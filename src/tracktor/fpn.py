# FPN modified to be used in the tracker

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from fpn.model.fpn import resnet
from fpn.model.utils.config import cfg
from torch.autograd import Variable


class FPN(resnet.resnet):

    # def forward(self, rois):
    #     net_conv = self._net_conv

    #     if frcnn_cfg.POOLING_MODE == 'crop':
    #         pool5 = self._crop_pool_layer(net_conv, rois)
    #     else:
    #         pool5 = self._roi_pool_layer(net_conv, rois)

    #     fc7 = self._head_to_tail(pool5)

    #     self._predictions["fc7"] = fc7

    #     cls_prob, bbox_pred = self._region_classification(fc7)

    #     stds = bbox_pred.data.new(frcnn_cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
    #     means = bbox_pred.data.new(frcnn_cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
    #     self._predictions["bbox_pred"] = bbox_pred.mul(Variable(stds)).add(Variable(means))

    #     cls_score, cls_prob, bbox_pred, rois = (self._predictions["cls_score"].data,
    #         self._predictions['cls_prob'].data,
    #         self._predictions['bbox_pred'].data,
    #         self._predictions['rois'].data[:,1:5])

    #     return cls_score, cls_prob, bbox_pred, rois

    def test_rois(self, rois):
        batch_size = self.im_data.size(0)
        rois = rois.cuda()
        padding = torch.zeros(rois.size(0), 1).cuda()
        rois_padd = torch.cat((padding, rois), 1)
        rois_padd = Variable(rois_padd, volatile=True)

        roi_pool_feat = self._PyramidRoI_Feat(
            self.mrcnn_feature_maps, rois_padd, self.im_info)

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(roi_pool_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, dim=1)

        rois_padd = rois_padd.view(batch_size, -1, rois_padd.size(1))
        cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1))
        bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if cfg.CLASS_AGNOSTIC_BBX_REG:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * self.n_classes)

        box_deltas = box_deltas.squeeze(dim=0)
        cls_score = cls_score.squeeze(dim=0).data
        cls_prob = cls_prob.squeeze(dim=0).data
        return cls_score, cls_prob, box_deltas, rois

    # def detect(self):
    #     net_conv = self._net_conv
    #     # build the anchors for the image
    #     self._anchor_component(net_conv.size(2), net_conv.size(3))
    #     rois = self._region_proposal(net_conv)
    #     return self.forward(rois)

    def load_image(self, image, im_info):
        self.im_data = Variable(image.permute(0, 3, 1, 2).cuda(), volatile=True)
        self.im_info = im_info.unsqueeze(dim=0).cuda()

        # feed image data to base model to obtain base feature map
        # Bottom-up
        c1 = self.RCNN_layer0(self.im_data)
        c2 = self.RCNN_layer1(c1)
        c3 = self.RCNN_layer2(c2)
        c4 = self.RCNN_layer3(c3)
        c5 = self.RCNN_layer4(c4)
        # Top-down
        p5 = self.RCNN_toplayer(c5)
        p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
        p4 = self.RCNN_smooth1(p4)
        p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
        p3 = self.RCNN_smooth2(p3)
        p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
        p2 = self.RCNN_smooth3(p2)

        # p6 = self.maxpool2d(p5)

        # rpn_feature_maps = [p2, p3, p4, p5, p6]
        self.mrcnn_feature_maps = [p2, p3, p4, p5]
        # pass
        # self.eval()
        # self._image = Variable(image.permute(0,3,1,2).cuda(), volatile=True)
        # self._im_info = im_info.cpu().numpy() # No need to change; actually it can be an list

        # self._mode = 'TEST'

        # # This is just _build_network in tf-faster-rcnn
        # torch.backends.cudnn.benchmark = False

        # self._net_conv = self._image_to_head()
