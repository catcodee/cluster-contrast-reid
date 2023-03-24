# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory, ClusterMemoryWithSoftLabel
from clustercontrast.trainers import ClusterContrastTrainerPCBSoft
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.utils.compute_cluster_dist import compute_cluster_dist
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type, num_stripes=args.num_stripes, 
                          stripe_pooling_type=args.stripe_pooling_type, num_stripe_features=args.stripe_features)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    num_stripes = args.num_stripes

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = []
    normal_params = []
    special_params = []
    for name, value in model.named_parameters():
        if value.requires_grad:
            if 'list' in name:
                special_params.append(value)
            else:
                normal_params.append(value)
    params.append({"params":normal_params, "lr":args.lr})
    params.append({"params":special_params, "lr":args.lr*2.0})
    # params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = ClusterContrastTrainerPCBSoft(model, alpha=args.alpha, theta=args.theta, use_hard_label=args.use_hard_label, use_local_label=args.use_local_label)

    for epoch in range(args.epochs):
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))

            features, _ = extract_features(model, cluster_loader, print_freq=50)
            global_features = torch.cat([features[f][0].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            local_features = torch.cat([features[f][1].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            local_features_list = torch.split(local_features, args.stripe_features, dim=1)
            if (((epoch+1) > args.soft_epoch) and args.use_local_label) or (args.use_local_dist and ((epoch+1) > args.soft_epoch)):
                local_rerank_dist_list = []
                for i in range(len(local_features_list)):
                    local_rerank_dist_list.append(compute_jaccard_distance(
                        local_features_list[i].contiguous(), k1=args.k1, k2=args.k2
                    ))
            global_rerank_dist = compute_jaccard_distance(global_features, k1=args.k1, k2=args.k2)

            if (args.use_local_dist and (epoch+1) > args.soft_epoch) or (args.use_local_label and (epoch+1) > args.soft_epoch):
                local_rerank_dist = np.zeros_like(global_rerank_dist)
                for i in range(args.num_stripes):
                    local_rerank_dist += local_rerank_dist_list[i]
                local_rerank_dist = local_rerank_dist / num_stripes
                if args.use_local_dist:
                    global_rerank_dist = (1-args.beta)*global_rerank_dist + args.beta*local_rerank_dist

            if epoch == 0:
                # DBSCAN cluster
                eps = args.eps
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

            # select & cluster images as training set of this epochs
            global_pseudo_labels = cluster.fit_predict(global_rerank_dist)
            global_num_cluster = len(set(global_pseudo_labels)) - (1 if -1 in global_pseudo_labels else 0)

            if ((epoch+1) > args.soft_epoch) and args.use_local_label:
                local_pseudo_labels_list = []
                local_num_cluster_list = []
                for i in range(num_stripes):
                    local_pseudo_labels = cluster.fit_predict(local_rerank_dist_list[i])
                    local_num_cluster = len(set(local_pseudo_labels)) - (1 if -1 in local_pseudo_labels else 0)
                    local_pseudo_labels_list.append(local_pseudo_labels)
                    local_num_cluster_list.append(local_num_cluster)
                label_transfer_matrix_list = []
                for i in range(args.num_stripes):
                    label_transfer_matrix_list.append(
                        compute_cluster_dist(local_pseudo_labels_list[i], global_pseudo_labels, local_num_cluster_list[i], global_num_cluster).cuda()
                    )
            else:
                local_pseudo_labels_list = [global_pseudo_labels]*num_stripes
                local_num_cluster_list = [global_num_cluster]*num_stripes
                label_transfer_matrix_list = [torch.eye(global_num_cluster).cuda()]*num_stripes

        trainer.label_transfer_matrix_list = label_transfer_matrix_list
            # print("epoch: {} \n pseudo_labels: {}".format(epoch, pseudo_labels.tolist()[:100]))

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_global_features = generate_cluster_features(global_pseudo_labels, global_features)
        cluster_local_features_list = []
        for i in range(num_stripes):
            cluster_local_features = generate_cluster_features(local_pseudo_labels_list[i], local_features_list[i])
            cluster_local_features_list.append(cluster_local_features)

        del cluster_loader, features, global_features, local_features_list, local_features

        # Create hybrid memory
        # if epoch < args.soft_epoch:
        #     global_memory = ClusterMemory(model.module.num_features, global_num_cluster, temp=args.temp,
        #                         momentum=args.momentum, use_hard=args.use_hard).cuda()
        #     global_memory.features = F.normalize(cluster_global_features, dim=1).cuda()

        #     local_memories = []
        #     for i in range(num_stripes):
        #         local_memory = ClusterMemory(model.module.num_stripe_features, global_num_cluster, temp=args.temp,
        #                 momentum=args.momentum, use_hard=args.use_hard).cuda()
        #         local_memory.features = F.normalize(cluster_local_features_list[i], dim=1).cuda()
        #         local_memories.append(local_memory)
        # else:
        global_memory = ClusterMemoryWithSoftLabel(model.module.num_features, global_num_cluster, temp=args.temp,
                            momentum=args.momentum, use_hard=args.use_hard).cuda()
        global_memory.features = F.normalize(cluster_global_features, dim=1).cuda()

        local_memories = []
        for i in range(num_stripes):
            local_memory = ClusterMemoryWithSoftLabel(model.module.num_stripe_features, local_num_cluster_list[i], temp=args.temp,
                    momentum=args.momentum, use_hard=args.use_hard).cuda()
            local_memory.features = F.normalize(cluster_local_features_list[i], dim=1).cuda()
            local_memories.append(local_memory)


        trainer.global_memory = global_memory
        trainer.local_memories = local_memories

        reserve_sample = torch.LongTensor(global_pseudo_labels) >= 0
        for i in range(num_stripes):
            reserve_sample = reserve_sample & (torch.LongTensor(local_pseudo_labels_list[i]) >= 0)
        pseudo_labeled_dataset = []
        labels_tensor = torch.stack([torch.LongTensor(global_pseudo_labels)] + [torch.LongTensor(l) for l in local_pseudo_labels_list], dim=0).T
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), labels_tensor)):
            if reserve_sample[i]:
                pseudo_labeled_dataset.append((fname, label, cid))

        print('==> Statistics for epoch {}: {} clusters'.format(epoch, global_num_cluster))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset, no_cam=args.no_cam)

        train_loader.new_epoch()

        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader))

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False, num_stripe=args.num_stripes, beta=args.beta)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, num_stripe=args.num_stripes, beta=args.beta)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='pcb',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--stripe-features', type=int, default=512)
    parser.add_argument('--num-stripes', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--stripe-pooling-type', type=str, default='gem')

    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--theta', type=float, default=0.95)

    parser.add_argument('--soft-epoch', type=int, default=5)
    parser.add_argument('--use-hard-label', action="store_true")
    parser.add_argument('--use-local-label', action="store_true")
    parser.add_argument('--use-local-dist', action="store_true")


    main()
