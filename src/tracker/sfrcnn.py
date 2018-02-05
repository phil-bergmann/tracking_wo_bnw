# Simple FRCNN for testing the simple tracker

from nets.vgg16 import vgg16
from model.config import cfg as frcnn_cfg

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F



class FRCNN(vgg16):
	
	
	def _predict(self, rois=None):
		# This is just _build_network in tf-faster-rcnn
		torch.backends.cudnn.benchmark = False
		net_conv = self._image_to_head()

		# build the anchors for the image
		self._anchor_component(net_conv.size(2), net_conv.size(3))

		if not hasattr(rois, 'size'):
			rois = self._region_proposal(net_conv)
		else:
			self._predictions["rois"] = rois

		if frcnn_cfg.POOLING_MODE == 'crop':
			pool5 = self._crop_pool_layer(net_conv, rois)
		else:
			pool5 = self._roi_pool_layer(net_conv, rois)

		if self._mode == 'TRAIN':
			torch.backends.cudnn.benchmark = True # benchmark because now the input size are fixed
		fc7 = self._head_to_tail(pool5)

		self._predictions["fc7"] = fc7

		cls_prob, bbox_pred = self._region_classification(fc7)
	
		#for k in self._predictions.keys():
		#	self._score_summaries[k] = self._predictions[k]

		return rois, cls_prob, bbox_pred

	def get_rois(self):
		return self._predictions["rois"]

	def get_fc7(self):
		return self._predictions["fc7"].data.cpu().numpy()

	def forward(self, image, im_info, rois=None, mode='TEST'):
		#self._image_gt_summaries['image'] = image
		#self._image_gt_summaries['gt_boxes'] = gt_boxes
		#self._image_gt_summaries['im_info'] = im_info

		self._image = Variable(torch.from_numpy(image.transpose([0,3,1,2])).cuda(), volatile=mode == 'TEST')
		self._im_info = im_info # No need to change; actually it can be an list

		self._mode = mode

		rois, cls_prob, bbox_pred = self._predict(rois)

		stds = bbox_pred.data.new(frcnn_cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
		means = bbox_pred.data.new(frcnn_cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
		self._predictions["bbox_pred"] = bbox_pred.mul(Variable(stds)).add(Variable(means))

	def test_image(self, image, im_info, rois=None):
		self.eval()
		self.forward(image, im_info, rois)
		cls_score, cls_prob, bbox_pred, rois = self._predictions["cls_score"].data.cpu().numpy(), \
													 self._predictions['cls_prob'].data.cpu().numpy(), \
													 self._predictions['bbox_pred'].data.cpu().numpy(), \
													 self._predictions['rois'].data.cpu().numpy()
		return cls_score, cls_prob, bbox_pred, rois



