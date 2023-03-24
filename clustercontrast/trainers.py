from __future__ import print_function, absolute_import
import time
import torch
from .utils.meters import AverageMeter

def one_hot(x, class_count):
	return torch.eye(class_count).cuda()[x,:]

class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)

            # forward
            f_out = self._forward(inputs)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)
            loss = self.memory(f_out, labels)

            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

class ClusterContrastTrainerPCB(object):
    def __init__(self, encoder, global_memory=None, local_memories=[], num_stripes=2, alpha=0.1):
        super(ClusterContrastTrainerPCB, self).__init__()
        self.encoder = encoder
        self.global_memory = global_memory
        self.local_memories = local_memories
        self.num_stripes = num_stripes
        self.alpha = alpha

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)

            # forward
            global_feat, local_feat = self._forward(inputs)
            local_feats = torch.split(local_feat, self.encoder.module.num_stripe_features, dim=1)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)
            global_loss = self.global_memory(global_feat, labels)
            local_loss = 0.0
            for j in range(self.num_stripes):
                local_loss += self.local_memories[j](local_feats[j], labels)
            local_loss = local_loss * (1.0 / self.num_stripes)

            loss = global_loss + self.alpha*local_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

class ClusterContrastTrainerPCBSoft(object):
    def __init__(self, encoder, global_memory=None, local_memories=[], num_stripes=2, 
                alpha=0.1, theta=0.05, use_hard_label=True, use_local_label=False):
        super(ClusterContrastTrainerPCBSoft, self).__init__()
        self.encoder = encoder
        self.global_memory = global_memory
        self.local_memories = local_memories
        self.num_stripes = num_stripes
        self.alpha = alpha
        self.theta = theta
        self.use_hard_label = use_hard_label
        self.use_local_label = use_local_label
        self.label_transfer_matrix_list = None
        # self.label_transfer_matrix_T_list = []

        # # for i in range(len(self.label_transfer_matrix_list)):
        # #     tf_mat = self.label_transfer_matrix_list[i]
        # #     tf_mat_norm = tf_mat / tf_mat.sum(dim=1, keep_dim=True)
        # #     tf_mat_T_norm = tf_mat.T / tf_mat.T.sum(dim=1, keep_dim=True)
        # #     self.label_transfer_matrix_list[i] = tf_mat_norm
        # #     self.label_transfer_matrix_T_list.append(tf_mat_T_norm)

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)

            # forward
            global_feat, local_feat = self._forward(inputs)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)
            if self.use_hard_label:
                loss = self._compute_loss_with_hard(global_feat, local_feat, labels)
            else:
                loss = self._compute_loss_with_soft(global_feat, local_feat, labels)

            loss = loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

    def _compute_loss_with_hard(self, global_feat, local_feat, labels):
        global_labels = labels[:, 0]
        one_hot_global_labels = one_hot(global_labels, self.global_memory.num_samples).cuda()

        local_labels_list = []
        one_hot_local_labels_list = []
        soft_global_labels = torch.zeros(global_feat.size(0), self.global_memory.num_samples).cuda()
        for i in range(self.num_stripes):
            local_labels = labels[:, 1+i]
            local_labels_list.append(local_labels)
            one_hot_local_labels = one_hot(local_labels, self.local_memories[i].num_samples)
            one_hot_local_labels_list.append(one_hot_local_labels)
            soft_global_labels += torch.einsum("mn,bm->bn", [self.label_transfer_matrix_list[i], one_hot_local_labels])
        soft_global_labels = soft_global_labels / self.num_stripes
        soft_global_labels = soft_global_labels / soft_global_labels.sum(dim=1, keepdim=True)
        final_soft_global_labels = self.theta*one_hot_global_labels + (1-self.theta)*soft_global_labels
        global_loss = self.global_memory(global_feat, hard_targets=global_labels, soft_targets=final_soft_global_labels)

        local_feats = torch.split(local_feat, self.encoder.module.num_stripe_features, dim=1)
        local_loss = 0.0
        for i in range(self.num_stripes):
            soft_local_labels = torch.einsum("mn,bn->bm", [self.label_transfer_matrix_list[i], one_hot_global_labels])
            soft_local_labels = soft_local_labels / soft_local_labels.sum(dim=1, keepdim=True)
            one_hot_local_labels = one_hot_local_labels_list[i]
            if self.use_local_label:
                final_soft_local_labels = self.theta*soft_local_labels + (1-self.theta)*one_hot_local_labels
            else:
                final_soft_local_labels = self.theta*one_hot_local_labels + (1-self.theta)*soft_local_labels
            local_loss += self.local_memories[i](
                                local_feats[i], hard_targets=local_labels_list[i], soft_targets=final_soft_local_labels)

        loss = global_loss + self.alpha*local_loss*(1.0/self.num_stripes)

        return loss

    def _compute_loss_with_soft(self, global_feat, local_feat, labels):
        global_probs = self.global_memory(global_feat, hard_targets=labels[:, 0], out_probs=True)
        local_feats = torch.split(local_feat, self.encoder.module.num_stripe_features, dim=1)
        local_probs_list = []
        local_labels_list = []
        for i in range(self.num_stripes):
            local_labels = labels[:, 1+i]
            local_labels_list.append(local_labels)
            local_probs_list.append(self.local_memories[i](local_feats[i], hard_targets=local_labels, out_probs=True))

        global_labels = labels[:, 0]
        one_hot_global_labels = one_hot(global_labels, self.global_memory.num_samples).cuda()

        one_hot_local_labels_list = []
        soft_global_labels = torch.zeros(global_feat.size(0), self.global_memory.num_samples).cuda()
        for i in range(self.num_stripes):
            one_hot_local_labels = one_hot(local_labels_list[i], self.local_memories[i].num_samples)
            one_hot_local_labels_list.append(one_hot_local_labels)
            soft_global_labels += torch.einsum("mn,bm->bn", [self.label_transfer_matrix_list[i], local_probs_list[i].detach()])
        soft_global_labels = soft_global_labels / self.num_stripes
        soft_global_labels = soft_global_labels / soft_global_labels.sum(dim=1, keepdim=True)
        final_soft_global_labels = self.theta*one_hot_global_labels + (1-self.theta)*soft_global_labels
        global_loss = self.global_memory(probs=global_probs, hard_targets=global_labels, soft_targets=final_soft_global_labels)

        local_loss = 0.0
        for i in range(self.num_stripes):
            soft_local_labels = torch.einsum("mn,bn->bm", [self.label_transfer_matrix_list[i], global_probs.detach()])
            soft_local_labels = soft_local_labels / soft_local_labels.sum(dim=1, keepdim=True)
            one_hot_local_labels = one_hot_local_labels_list[i]
            if self.use_local_label:
                final_soft_local_labels = self.theta*one_hot_local_labels + (1-self.theta)*soft_local_labels
            else:
                final_soft_local_labels = self.theta*soft_local_labels + (1-self.theta)*one_hot_local_labels
            local_loss += self.local_memories[i](
                                probs=local_probs_list[i], hard_targets=local_labels_list[i], soft_targets=final_soft_local_labels)

        loss = global_loss + self.alpha*local_loss*(1.0/self.num_stripes)

        return loss

class CameraClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(CameraClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, cams, indexes = self._parse_data(inputs)

            # forward
            f_out = self._forward(inputs)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)
            loss = self.memory(f_out, labels, cams)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, cams, indexes = inputs
        return imgs.cuda(), pids.cuda(), cams.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


class SSClusterContrastTrainer(object):
    def __init__(self, encoder, dis_loss,memory=None):
        super(SSClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory

        self.dis_loss = dis_loss

    def train(self, epoch, data_loader, data_loader_s, optimizer, optimzier_s, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        losses_s = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)
            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)
            # forward
            f_out = self._forward(inputs)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)
            loss = self.memory(f_out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            inputs = data_loader_s.next()
            inputs = inputs[0]
            inputs = [data.cuda() for data in inputs]
            feat, mean_feat = self.encoder(inputs, mode=1)
            loss_s = self.dis_loss(feat, mean_feat)
            
            optimzier_s.zero_grad()
            loss_s.backward()
            optimzier_s.step()

            losses_s.update(loss_s.item())
            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss_s {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_s.val, losses_s.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

class SSCameraClusterContrastTrainer(object):
    def __init__(self, encoder, dis_loss,memory=None):
        super(SSCameraClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory

        self.dis_loss = dis_loss

    def train(self, epoch, data_loader, data_loader_s, optimizer, optimzier_s, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        losses_s = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)
            # process inputs
            inputs, labels, cams, indexes = self._parse_data(inputs)
            # forward
            f_out = self._forward(inputs)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)
            loss = self.memory(f_out, labels, cams)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _compute_loss_with_hard(self, global_feat, local_feat, labels):
        global_labels = labels[:, 0]
        one_hot_global_labels = one_hot(global_labels, self.global_memory.num_samples).cuda()

        local_labels_list = []
        one_hot_local_labels_list = []
        soft_global_labels = torch.zeros(global_feat.size(0), self.global_memory.num_samples).cuda()
        for i in range(self.num_stripes):
            local_labels = labels[:, 1+i]
            local_labels_list.append(local_labels)
            one_hot_local_labels = one_hot(local_labels, self.local_memories[i].num_samples)
            one_hot_local_labels_list.append(one_hot_local_labels)
            soft_global_labels += torch.einsum("mn,bm->bn", [self.label_transfer_matrix_list[i], one_hot_local_labels])
        soft_global_labels = soft_global_labels / self.num_stripes
        soft_global_labels = soft_global_labels / soft_global_labels.sum(dim=1, keepdim=True)
        final_soft_global_labels = self.theta*one_hot_global_labels + (1-self.theta)*soft_global_labels
        global_loss = self.global_memory(global_feat, hard_targets=global_labels, soft_targets=final_soft_global_labels)

        local_feats = torch.split(local_feat, self.encoder.module.num_stripe_features, dim=1)
        local_loss = 0.0
        for i in range(self.num_stripes):
            soft_local_labels = torch.einsum("mn,bn->bm", [self.label_transfer_matrix_T_list[i], one_hot_global_labels])
            soft_local_labels = soft_local_labels / soft_local_labels.sum(dim=1, keepdim=True)
            one_hot_local_labels = one_hot_local_labels_list[i]
            if self.use_local_label:
                final_soft_local_labels = self.theta*soft_local_labels + (1-self.theta)*one_hot_local_labels
            else:
                final_soft_local_labels = self.theta*one_hot_local_labels + (1-self.theta)*soft_local_labels
            local_loss += self.local_memories[i](
                                local_feats[i], hard_targets=local_labels_list[i], soft_targets=final_soft_local_labels)

        loss = global_loss + self.alpha*local_loss*(1.0/self.num_stripes)

        return loss

    def _compute_loss_with_soft(self, global_feat, local_feat, labels):
        global_probs = self.global_memory(global_feat, hard_targets=labels[:, 0], out_probs=True)
        local_feats = torch.split(local_feat, self.encoder.module.num_stripe_features, dim=1)
        local_probs_list = []
        local_labels_list = []
        for i in range(self.num_stripes):
            local_labels = labels[:, 1+i]
            local_labels_list.append(local_labels)
            local_probs_list.append(self.local_memories[i](local_feats[i], hard_targets=local_labels, out_probs=True))

        global_labels = labels[:, 0]
        one_hot_global_labels = one_hot(global_labels, self.global_memory.num_samples).cuda()

        one_hot_local_labels_list = []
        soft_global_labels = torch.zeros(global_feat.size(0), self.global_memory.num_samples).cuda()
        for i in range(self.num_stripes):
            one_hot_local_labels = one_hot(local_labels_list[i], self.local_memories[i].num_samples)
            one_hot_local_labels_list.append(one_hot_local_labels)
            soft_global_labels += torch.einsum("mn,bm->bn", [self.label_transfer_matrix_list[i], local_probs_list[i].detach()])
        soft_global_labels = soft_global_labels / self.num_stripes
        soft_global_labels = soft_global_labels / soft_global_labels.sum(dim=1, keepdim=True)
        final_soft_global_labels = self.theta*one_hot_global_labels + (1-self.theta)*soft_global_labels
        global_loss = self.global_memory(probs=global_probs, hard_targets=global_labels, soft_targets=final_soft_global_labels)

        local_loss = 0.0
        for i in range(self.num_stripes):
            soft_local_labels = torch.einsum("mn,bn->bm", [self.label_transfer_matrix_list[i], global_probs.detach()])
            soft_local_labels = soft_local_labels / soft_local_labels.sum(dim=1, keepdim=True)
            one_hot_local_labels = one_hot_local_labels_list[i]
            if self.use_local_label:
                final_soft_local_labels = self.theta*one_hot_local_labels + (1-self.theta)*soft_local_labels
            else:
                final_soft_local_labels = self.theta*soft_local_labels + (1-self.theta)*one_hot_local_labels
            local_loss += self.local_memories[i](
                                probs=local_probs_list[i], hard_targets=local_labels_list[i], soft_targets=final_soft_local_labels)

        loss = global_loss + self.alpha*local_loss*(1.0/self.num_stripes)

        return loss


