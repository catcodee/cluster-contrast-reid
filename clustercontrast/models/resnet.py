from __future__ import absolute_import
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from .pooling import build_pooling_layer


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'pcb']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, pooling_type='avg'):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        self.gap = build_pooling_layer(pooling_type)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x):
        bs = x.size(0)
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if (self.training is False):
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x

        return prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class PCB(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }
    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, num_stripe_features=0, norm=False, dropout=0, num_classes=0, 
                 pooling_type='avg', num_stripes=2, stripe_pooling_type='avg'):
        super(PCB, self).__init__()
        assert num_stripes > 1, "pcb的分块数小于2"
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in PCB.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = PCB.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

        self.gap = build_pooling_layer(pooling_type)
        self.stripe_pooling = build_pooling_layer(stripe_pooling_type)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.num_stripe_features = num_stripe_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.stripe_has_embedding = num_stripe_features > 0
            self.num_classes = num_classes
            self.num_stripes = num_stripes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                self.relu = nn.ReLU()
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)

            init.constant_(self.feat_bn.weight, 1)
            init.constant_(self.feat_bn.bias, 0)
            self.feat_bn.bias.requires_grad_(False)

            if self.stripe_has_embedding:
                self.feat_embedding_list = nn.ModuleList()
                for _ in range(self.num_stripes):
                    feat = nn.Linear(out_planes, self.num_stripe_features)
                    feat_bn = nn.BatchNorm1d(self.num_stripe_features)
                    init.kaiming_normal_(feat.weight, mode='fan_out')
                    init.constant_(feat.bias, 0)
                    init.constant_(feat_bn.weight, 1)
                    init.constant_(feat_bn.bias, 0)
                    feat_bn.bias.requires_grad_(False)
                    self.feat_embedding_list.append(
                        nn.Sequential(feat, feat_bn)
                    )
            else:
                self.feat_embedding_list = nn.ModuleList()
                # Change the num_features to CNN output channels
                self.num_stripe_features = out_planes
                for _ in range(self.num_stripes):
                    feat_bn = nn.BatchNorm1d(self.num_features)
                    init.constant_(feat_bn.weight, 1)
                    init.constant_(feat_bn.bias, 0)
                    feat_bn.bias.requires_grad_(False)
                    self.feat_embedding_list.append(feat_bn)

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(out_planes, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)

                for _ in num_stripes:
                    self.stripe_classifiers = nn.ModuleList()
                    if self.num_stripe_features > 0:
                        classifier = nn.Linear(self.num_stripe_features, self.num_classes, bias=False)
                    else:
                        classifier = nn.Linear(out_planes, self.num_classes, bias=False)
                    init.normal_(classifier, std=0.001)
                    self.stripe_classifiers.append(classifier)

        if not pretrained:
            self.reset_params()

    def forward(self, x):
        bs = x.size(0)
        x = self.base(x)
        
        stripe_h = int(x.shape[2] // self.num_stripes)
        stripes = torch.split(x, stripe_h, dim=2)
        stripe_feats = []
        for i in range(self.num_stripes):
            feat = self.stripe_pooling(stripes[i])
            feat = feat.view(feat.size(0), -1)
            stripe_feats.append(feat)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            stripe_feat = torch.cat(stripe_feats, dim=1)
            return x, stripe_feat

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if self.stripe_has_embedding:
            for i in range(self.num_stripes):
                stripe_feats[i] = self.feat_embedding_list[i](stripe_feats[i])

        if (self.training is False):
            bn_x = F.normalize(bn_x)
            for i in range(self.num_stripes):
                stripe_feats[i] = F.normalize(stripe_feats[i])
            stripe_feat = torch.cat(stripe_feats, dim=1)
            return bn_x, stripe_feat

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.norm:
            for i in range(self.num_stripes):
                stripe_feats[i] = F.normalize(stripe_feats[i])
        elif self.stripe_has_embedding:
            for i in range(self.num_stripes):
                stripe_feats[i] = F.relu(stripe_feats[i])

        if self.dropout > 0:
            bn_x = self.drop(bn_x)
            for i in range(self.num_stripes):
                stripe_feats[i] = self.drop(stripe_feats[i])

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
            stripe_probs = []
            for i in range(self.num_stripes):
                stripe_probs.append(
                    self.stripe_classifiers[i](stripe_feats[i])
                )
        else:
            stripe_feat = torch.cat(stripe_feats, dim=1)
            return bn_x, stripe_feat

        return prob, stripe_probs

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)

def pcb(**kwargs):
    return PCB(50, **kwargs)
