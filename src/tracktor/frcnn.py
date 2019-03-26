# FRCNN modified to be used in the tracker

from frcnn.nets.resnet_v1 import resnetv1
from frcnn.model.config import cfg as frcnn_cfg

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F


class FRCNN(resnetv1):

    def forward(self, rois):
        net_conv = self._net_conv

        if frcnn_cfg.POOLING_MODE == 'crop':
            pool5 = self._crop_pool_layer(net_conv, rois)
        else:
            pool5 = self._roi_pool_layer(net_conv, rois)

        fc7 = self._head_to_tail(pool5)

        self._predictions["fc7"] = fc7

        cls_prob, bbox_pred = self._region_classification(fc7)

        stds = bbox_pred.data.new(frcnn_cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
        means = bbox_pred.data.new(frcnn_cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
        self._predictions["bbox_pred"] = bbox_pred.mul(Variable(stds)).add(Variable(means))

        cls_score, cls_prob, bbox_pred, rois = (self._predictions["cls_score"].data,
            self._predictions['cls_prob'].data,
            self._predictions['bbox_pred'].data,
            self._predictions['rois'].data[:,1:5])

        return cls_score, cls_prob, bbox_pred, rois

    def test_rois(self, rois):
        rois_score = rois.new(rois.size()[0],1).zero_()
        rois = torch.cat((rois_score, rois), 1)
        rois = Variable(rois).cuda()
        self._predictions["rois"] = rois
        return self.forward(rois)

    def detect(self):
        net_conv = self._net_conv
        # build the anchors for the image
        self._anchor_component(net_conv.size(2), net_conv.size(3))
        rois = self._region_proposal(net_conv)
        return self.forward(rois)

    def load_image(self, image, im_info):
        self.eval()
        self._image = Variable(image.permute(0,3,1,2).cuda(), volatile=True)
        self._im_info = im_info.cpu().numpy() # No need to change; actually it can be an list

        self._mode = 'TEST'

        # This is just _build_network in tf-faster-rcnn
        torch.backends.cudnn.benchmark = False

        self._net_conv = self._image_to_head()
