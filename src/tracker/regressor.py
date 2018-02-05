# Regressor that tries to keep the features the same

from model.bbox_transform import bbox_transform, bbox_transform_inv
from model.test import _clip_boxes

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Regressor(nn.Module):

	def __init__(self, num_classes=2, frcnn=None):
		super(Regressor, self).__init__()

		self._num_classes = num_classes
		self.frcnn = frcnn
		self.bbox_pred_net = nn.Linear(4096, self._num_classes * 4)

		self.init_weights()

	def forward(self, input):
		"""Input should be fc7"""

		return self.bbox_pred_net(input)

	def val_predict(self, blobs):
		tracks = blobs['tracks']
		old_pos = Variable(torch.from_numpy(tracks[:,0,:])).cuda()
		targets = Variable(torch.from_numpy(tracks[:,1,:])).cuda()

		rois_score = Variable(torch.zeros(tracks.shape[0],1)).cuda()
		rois_bb = old_pos
		rois_old = torch.cat((rois_score, rois_bb),1)

		_, _, _, _ = self.frcnn.test_image(blobs['data'][1], blobs['im_info'], rois_old)
		fc7 = self.frcnn.get_fc7()

		bbox_pred = self.forward(Variable(torch.from_numpy(fc7)).cuda())

		boxes = bbox_transform_inv(old_pos, bbox_pred).cpu().data.numpy()
		boxes = _clip_boxes(boxes, blobs['im_info'][0,:2])

		# class human
		cl = 1
		boxes = boxes[:, cl*4:(cl+1)*4]

		t = np.stack((tracks[:,0,:],boxes), 1)

		return Variable(torch.from_numpy(t)).cuda()

	def sum_losses(self, blobs):
		self._losses = {}

		assert self.frcnn != None, "[!] No FRCNN found!"

		tracks = blobs['tracks']
		old_pos = Variable(torch.from_numpy(tracks[:,0,:])).cuda()
		targets = Variable(torch.from_numpy(tracks[:,1,:])).cuda()

		rois_score = Variable(torch.zeros(tracks.shape[0],1)).cuda()
		rois_bb = old_pos
		rois_old = torch.cat((rois_score, rois_bb),1)

		_, _, _, _ = self.frcnn.test_image(blobs['data'][1], blobs['im_info'], rois_old)
		fc7 = self.frcnn.get_fc7()

		# Calculate the regression loss
		bbox_targets = bbox_transform(old_pos, targets)

		bbox_pred = self.forward(Variable(torch.from_numpy(fc7)).cuda())

		loss_box = self.smooth_L1(bbox_pred[:,4:], bbox_targets)

		boxes = bbox_transform_inv(old_pos, bbox_pred).cpu().data.numpy()
		boxes = _clip_boxes(boxes, blobs['im_info'][0,:2])

		# class human
		cl = 1
		boxes = boxes[:, cl*4:(cl+1)*4]
		rois_score = Variable(torch.zeros(boxes.shape[0],1)).cuda()
		rois_bb = Variable(torch.from_numpy(boxes)).cuda()
		rois_old = torch.cat((rois_score, rois_bb),1)
		_, _, _, _ = self.frcnn.test_image(blobs['data'][1], blobs['im_info'], rois_old)

		fc7_new = self.frcnn.get_fc7()

		# fc7 score
		loss_fc7 = Variable(torch.from_numpy(np.mean(np.abs(fc7-fc7_new),1))).cuda().mean()

		# Calculate the feature loss

		#total_loss = loss_box + loss_fc7
		total_loss = loss_box

		self._losses['total_loss'] = total_loss
		self._losses['loss_box'] = loss_box
		self._losses['loss_fc7'] = loss_fc7

		return self._losses

	def smooth_L1(self, bbox_pred, bbox_targets):
		box_diff = bbox_pred - bbox_targets
		abs_box_diff = torch.abs(box_diff)

		smoothL1_sign = (abs_box_diff < 1.).detach().float()


		loss_box = torch.pow(abs_box_diff, 2) / 2. * smoothL1_sign \
					+ (abs_box_diff - 0.5) * (1. - smoothL1_sign)

		loss_box = loss_box.sum(1)

		loss_box = loss_box.mean()

		return loss_box


	def init_weights(self):
		"""Initialization function from FRCNN"""
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
		normal_init(self.bbox_pred_net, 0, 0.001, False)

	def load_state_dict(self, state_dict):
		"""
		Load only what is available here, useful for initialization with frcnn regressor
		"""

		# Only load bbox_pred_net
		load = ['bbox_pred_net.bias', 'bbox_pred_net.weight']

		pretrained_state_dict = {k: v for k,v in state_dict.items() for nk in load if k==nk}
		updated_state_dict = self.state_dict()
		updated_state_dict.update(pretrained_state_dict)
		nn.Module.load_state_dict(self, updated_state_dict)

		#nn.Module.load_state_dict(self, {k: state_dict[k] for k in list(self.state_dict())})
		#print(self.bbox_pred_net.state_dict())
		# Throw away any entry that doesn't match in shape to actual model
		#pretrained_state_dict = {k: v for k,v in state_dict.items() for nk,nv in self.state_dict().items() if k==nk if v.size()==nv.size()}
		# Update old state dict, so all parameters are present when loading
		#state_dict = self.state_dict()
		#print(pretrained_state_dict)
		#state_dict.update(pretrained_state_dict)
		#self.load_state_dict(state_dict)
		#print(pretrained_state_dict)