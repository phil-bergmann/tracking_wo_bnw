# Simple FRCNN for testing the simple tracker

from nets.resnet_v1 import resnetv1
from model.config import cfg as frcnn_cfg

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F



class FRCNN(resnetv1):
	
	
	def _predict(self, rois, image_to_head):
		# This is just _build_network in tf-faster-rcnn
		torch.backends.cudnn.benchmark = False

		if image_to_head:
			net_conv = self._image_to_head()
			self._net_conv = net_conv
		else:
			net_conv = self._net_conv

		if not hasattr(rois, 'size'):
			# build the anchors for the image
			self._anchor_component(net_conv.size(2), net_conv.size(3))
			rois = self._region_proposal(net_conv)
		else:
			rois = Variable(bb2rois(rois)).cuda()
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
		return self._predictions["rois"].data

	def get_fc7(self):
		return self._predictions["fc7"].data

	def forward(self, image, im_info, rois, mode='TEST'):
		#self._image_gt_summaries['image'] = image
		#self._image_gt_summaries['gt_boxes'] = gt_boxes
		#self._image_gt_summaries['im_info'] = im_info

		# should be changed later but for now it is ok
		image_to_head = False
		if hasattr(image, 'size'):
			self._image = Variable(torch.from_numpy(image.cpu().numpy().transpose([0,3,1,2])).cuda(), volatile=mode == 'TEST')
			self._im_info = im_info.cpu().numpy() # No need to change; actually it can be an list
			image_to_head = True

		self._mode = mode

		rois, cls_prob, bbox_pred = self._predict(rois, image_to_head)

		stds = bbox_pred.data.new(frcnn_cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
		means = bbox_pred.data.new(frcnn_cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
		self._predictions["bbox_pred"] = bbox_pred.mul(Variable(stds)).add(Variable(means))

	def test_image(self, image, im_info, rois=None):
		self.eval()
		self.forward(image, im_info, rois)
		cls_score, cls_prob, bbox_pred, rois = self._predictions["cls_score"].data, \
													 self._predictions['cls_prob'].data, \
													 self._predictions['bbox_pred'].data, \
													 self._predictions['rois'].data[:,1:5]
		return cls_score, cls_prob, bbox_pred, rois

	def test_rois(self, rois):
		return self.test_image(None, None, rois)

	def get_net_conv(self, image, im_info):
		self.eval()
		self._image = Variable(torch.from_numpy(image.cpu().numpy().transpose([0,3,1,2])).cuda(), volatile=True)
		self._im_info = im_info.cpu().numpy() # No need to change; actually it can be an list
		self._mode = 'TEST'
		torch.backends.cudnn.benchmark = False
		net_conv = self._image_to_head()
		return net_conv.data.cpu()

def boxes2rois(boxes, cl=1):
	rois_score = boxes.new(boxes.size()[0],1).zero_()
	rois_bb = boxes[:, cl*4:(cl+1)*4]
	rois = torch.cat((rois_score, rois_bb),1)

	return rois

def bb2rois(bb):
	rois_score = bb.new(bb.size()[0],1).zero_()
	rois = torch.cat((rois_score, bb), 1)
	return rois
