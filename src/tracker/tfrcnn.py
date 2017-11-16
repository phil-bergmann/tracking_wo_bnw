from nets.vgg16 import vgg16
from model.config import cfg as frcnn_cfg

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


class tfrcnn(vgg16):

	# Not all modules are needed anymore
	def _init_modules(self):
		self.vgg = models.vgg16()
		# Remove fc8
		self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier._modules.values())[:-1])

		# Fix the layers before conv3:
		for layer in range(10):
			for p in self.vgg.features[layer].parameters(): p.requires_grad = False

		# not using the last maxpool layer
		self._layers['head'] = nn.Sequential(*list(self.vgg.features._modules.values())[:-1])

		# rpn
		self.rpn_net = nn.Conv2d(512, 512, [3, 3], padding=1)

		self.rpn_cls_score_net = nn.Conv2d(512, self._num_anchors * 2, [1, 1])
	
		self.rpn_bbox_pred_net = nn.Conv2d(512, self._num_anchors * 4, [1, 1])

		#######
		self.cls_score_net = nn.Linear(4096, self._num_classes)
    	#self.bbox_pred_net = nn.Linear(4096, self._num_classes * 4)
    	#######

		self.init_weights()

	def init_weights(self):
		def normal_init(m, mean, stddev, truncated=False):
			"""
			weight initalizer: truncated normal and random normal.
			"""
			# x is a parameter
			if truncated:
				m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
			else:
				m.weight.data.normal_(mean, stddev)
			m.bias.data.zero_()

		normal_init(self.rpn_net, 0, 0.01, frcnn_cfg.TRAIN.TRUNCATED)
		normal_init(self.rpn_cls_score_net, 0, 0.01, frcnn_cfg.TRAIN.TRUNCATED)
		normal_init(self.rpn_bbox_pred_net, 0, 0.01, frcnn_cfg.TRAIN.TRUNCATED)

	

	def _predict(self):
		# This is just _build_network in tf-faster-rcnn
		torch.backends.cudnn.benchmark = False
		net_conv = self._image_to_head()

		# build the anchors for the image
		self._anchor_component(net_conv.size(2), net_conv.size(3))

   		# rois is a 300x5 vector (img_index,x1,y1,x2,y2), img_index = 0
		rois = self._region_proposal(net_conv)
		if frcnn_cfg.POOLING_MODE == 'crop':
			pool5 = self._crop_pool_layer(net_conv, rois)
		else:
			pool5 = self._roi_pool_layer(net_conv, rois)

		if self._mode == 'TRAIN':
			torch.backends.cudnn.benchmark = True # benchmark because now the input size are fixed
		fc7 = self._head_to_tail(pool5)

		self._predictions["fc7"] = fc7

		# Not needed, only for the moment
		self._region_classification(fc7)
	
		#for k in self._predictions.keys():
		#	self._score_summaries[k] = self._predictions[k]

		#return rois

	def _region_classification(self, fc7):
    	cls_score = self.cls_score_net(fc7)
    	cls_pred = torch.max(cls_score, 1)[1]
    	cls_prob = F.softmax(cls_score)
    	#bbox_pred = self.bbox_pred_net(fc7)

    	self._predictions["cls_score"] = cls_score
    	self._predictions["cls_pred"] = cls_pred
    	self._predictions["cls_prob"] = cls_prob
    	#self._predictions["bbox_pred"] = bbox_pred

	def forward(self, image, im_info, gt_boxes=None, mode='TRAIN'):
		#self._image_gt_summaries['image'] = image
		#self._image_gt_summaries['gt_boxes'] = gt_boxes
		#self._image_gt_summaries['im_info'] = im_info

		self._image = Variable(torch.from_numpy(image.transpose([0,3,1,2])).cuda(), volatile=mode == 'TEST')
		self._im_info = im_info # No need to change; actually it can be an list
		#self._gt_boxes = Variable(torch.from_numpy(gt_boxes).cuda()) if gt_boxes is not None else None

		self._mode = mode

		self._predict()


	def test_image(self, image, im_info):
		"""Tests a image
		returns a torch variable
		"""
		self.eval()
		self.forward(image, im_info, None, mode='TEST')
		rois, fc7, scores = self._predictions['rois'].detach(), self._predictions['fc7'].detach(), self._predictions["cls_prob"].detach()
		return rois, fc7, scores



