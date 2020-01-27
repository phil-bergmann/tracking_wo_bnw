import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import Bottleneck
import torchvision.models as models
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor

import numpy as np
import random
import cv2
import math

from .triplet_loss import _get_anchor_positive_triplet_mask, _get_anchor_negative_triplet_mask, _get_triplet_mask

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResNet(models.ResNet):
    def __init__(self, block, layers, output_dim):
        super(ResNet, self).__init__(block, layers)
        
        self.name = "ResNet"

        self.avgpool = nn.AvgPool2d((8,4), stride=1)
        self.fc = nn.Linear(512 * block.expansion, 1024)
        self.bn_fc = nn.BatchNorm1d(1024)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(1024, output_dim)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.fc_compare = nn.Linear(output_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn_fc(x)
        x = self.relu_fc(x)
        x = self.fc_out(x)

        return x

    def test_rois(self, image, rois):
        """Tests the rois on a particular image. Should be inside image."""
        x = self.build_crops(image, rois)
        x = Variable(x)
        
        return self.forward(x)

    def compare(self, e0, e1, train=False):
        out = torch.abs(e0 - e1)
        out = self.fc_compare(out)
        if not train:
            out = torch.sigmoid(out)
        return out

    def build_crops(self, image, rois):
        res = []
        trans = Compose([ToPILImage(), Resize((256,128)), ToTensor()])
        for r in rois:
            x0 = int(r[0])
            y0 = int(r[1])
            x1 = int(r[2])
            y1 = int(r[3])
            if x0 == x1:
                if x0 != 0:
                    x0 -= 1
                else:
                    x1 += 1
            if y0 == y1:
                if y0 != 0:
                    y0 -= 1
                else:
                    y1 += 1
            im = image[0,:,y0:y1,x0:x1]
            im = trans(im)
            res.append(im)
        res = torch.stack(res, 0)
        res = res.cuda()
        return res

    def sum_losses(self, batch, loss, margin, prec_at_k):
        """For Pretraining

        Function for preatrainind this CNN with the triplet loss. Takes a sample of N=PK images, P different
        persons, K images of each. K=4 is a normal parameter.

        [!] Batch all and batch hard should work fine. Take care with weighted triplet or cross entropy!!

        Args:
            batch (list): [images, labels], images are Tensor of size (N,H,W,C), H=224, W=112, labels Tensor of
            size (N)
        """

        inp = batch[0][0]
        inp = Variable(inp).cuda()

        labels = batch[1][0]
        labels = labels.cuda()

        embeddings = self.forward(inp)
        
        if loss == "cross_entropy":
            m = _get_triplet_mask(labels).nonzero()
            e0 = []
            e1 = []
            e2 = []
            for p in m:
                e0.append(embeddings[p[0]])
                e1.append(embeddings[p[1]])
                e2.append(embeddings[p[2]])
            e0 = torch.stack(e0,0)
            e1 = torch.stack(e1,0)
            e2 = torch.stack(e2,0)

            out_pos = self.compare(e0, e1, train=True)
            out_neg = self.compare(e0, e2, train=True)

            tar_pos = Variable(torch.ones(out_pos.size(0)).view(-1,1).cuda())
            tar_neg = Variable(torch.zeros(out_pos.size(0)).view(-1,1).cuda())

            loss_pos = F.binary_cross_entropy_with_logits(out_pos, tar_pos)
            loss_neg = F.binary_cross_entropy_with_logits(out_neg, tar_neg)

            total_loss = (loss_pos + loss_neg)/2

        elif loss == 'batch_all':
            # works, batch all strategy
            m = _get_triplet_mask(labels).nonzero()
            e0 = []
            e1 = []
            e2 = []
            for p in m:
                e0.append(embeddings[p[0]])
                e1.append(embeddings[p[1]])
                e2.append(embeddings[p[2]])
            e0 = torch.stack(e0,0)
            e1 = torch.stack(e1,0)
            e2 = torch.stack(e2,0)
            total_loss = F.triplet_margin_loss(e0, e1, e2, margin=margin, p=2)
        elif loss == 'batch_hard':
            # compute pariwise square distance matrix, not stable with sqr as 0 can happen
            n = embeddings.size(0)
            m = embeddings.size(0)
            d = embeddings.size(1)

            x = embeddings.data.unsqueeze(1).expand(n, m, d)
            y = embeddings.data.unsqueeze(0).expand(n, m, d)

            dist = torch.pow(x - y, 2).sum(2)

            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

            pos_dist = dist * mask_anchor_positive
            # here add value so that not valid values can not be picked
            max_val = torch.max(dist)
            neg_dist = dist + max_val * (1.0 - mask_anchor_negative)

            # for each anchor compute hardest pair
            triplets = []
            for i in range(dist.size(0)):
                pos = torch.max(pos_dist[i],0)[1].item()
                neg = torch.min(neg_dist[i],0)[1].item()
                triplets.append((i, pos, neg))

            e0 = []
            e1 = []
            e2 = []
            for p in triplets:
                e0.append(embeddings[p[0]])
                e1.append(embeddings[p[1]])
                e2.append(embeddings[p[2]])
            e0 = torch.stack(e0,0)
            e1 = torch.stack(e1,0)
            e2 = torch.stack(e2,0)
            total_loss = F.triplet_margin_loss(e0, e1, e2, margin=margin, p=2)

        elif loss == 'weighted_triplet':
            # compute pairwise distance matrix
            dist = []
            # iteratively construct the columns
            for e in embeddings:
                ee = torch.cat([e.view(1,-1) for _ in range(embeddings.size(0))],0)
                dist.append(F.pairwise_distance(embeddings, ee))
            dist = torch.cat(dist, 1)

            # First, we need to get a mask for every valid positive (they should have same label)
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
            pos_dist = dist * Variable(mask_anchor_positive.float())

            # Now every valid negative mask
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
            neg_dist = dist * Variable(mask_anchor_negative.float())

            # now get the weights for each anchor, detach because it should be a constant weighting factor
            pos_weights = Variable(torch.zeros(dist.size()).cuda())
            neg_weights = Variable(torch.zeros(dist.size()).cuda())
            for i in range(dist.size(0)):
                # make by line
                mask = torch.zeros(dist.size()).byte().cuda()
                mask[i] = 1
                pos_weights[mask_anchor_positive & mask] = F.softmax(pos_dist[mask_anchor_positive & mask], 0)
                neg_weights[mask_anchor_negative & mask] = F.softmin(neg_dist[mask_anchor_negative & mask], 0)
            pos_weights = pos_weights.detach()
            neg_weights = neg_weights.detach()
            pos_weight_dist = pos_dist * pos_weights
            neg_weight_dist = neg_dist * neg_weights

            triplet_loss = torch.clamp(margin + pos_weight_dist.sum(1, keepdim=True) - neg_weight_dist.sum(1, keepdim=True), min=0)
            total_loss = triplet_loss.mean()
        else:
            raise NotImplementedError("Loss: {}".format(loss))

        losses = {}

        if prec_at_k:
            # compute pariwise square distance matrix, not stable with sqr as 0 can happen
            n = embeddings.size(0)
            m = embeddings.size(0)
            d = embeddings.size(1)

            x = embeddings.data.unsqueeze(1).expand(n, m, d)
            y = embeddings.data.unsqueeze(0).expand(n, m, d)

            dist = torch.pow(x - y, 2).sum(2)
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
            _, indices = torch.sort(dist, dim=1)
            num_hit = 0
            num_ges = 0
            for i in range(dist.size(0)):
                d = mask_anchor_positive[i].nonzero().view(-1,1)
                ind = indices[i][:prec_at_k+1]

                same = d==ind
                num_hit += same.sum()
                num_ges += prec_at_k
            k_loss = torch.Tensor(1)
            k_loss[0] = num_hit / num_ges
            losses['prec_at_k'] = Variable(k_loss.cuda())

        losses['total_loss'] = total_loss

        return losses

    def load_pretrained_dict(self, state_dict):
        """Load the pretrained weights and ignore the ones where size does not match"""
        pretrained_state_dict = {k: v for k,v in state_dict.items() for kk,vv in self.state_dict().items() if k==kk and v.size() == vv.size()}
        updated_state_dict = self.state_dict()
        updated_state_dict.update(pretrained_state_dict)
        self.load_state_dict(updated_state_dict)


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_pretrained_dict(model_zoo.load_url(model_urls['resnet50']))
    return model