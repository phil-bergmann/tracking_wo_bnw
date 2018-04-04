import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np

from model.bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes

class Simple_Regressor(nn.Module):

	def __init__(self, hidden_dim, num_layers):
		super(Simple_Regressor, self).__init__()
		self.name = "Simple_Regressor"

		self.hidden_dim = hidden_dim
		self.input_dim = 4096
		self.num_layers = num_layers

		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

		self.out = nn.Linear(self.hidden_dim, 4)

	def init_hidden(self, minibatch_size=1):
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (Variable(torch.zeros(self.num_layers, minibatch_size, self.hidden_dim)).cuda(),
				Variable(torch.zeros(self.num_layers, minibatch_size, self.hidden_dim)).cuda())

	def init_weights(self):

		# initialize forget gates to 1
		for names in self.lstm._all_weights:
			for name in filter(lambda n: "bias" in n,  names):
				bias = getattr(self.lstm, name)
				n = bias.size(0)
				start, end = n//4, n//2
				bias.data[start:end].fill_(1.)

	def forward(self, inp, hidden):
		# Expects the input to be (seq_len, batch, input_size)
		# Output of lstm (seq_len, batch, hidden_size)
		lstm_out, hidden = self.lstm(inp.view(1,-1,self.input_dim), hidden)

		out = self.out(lstm_out[0,:,:])

		return out, hidden


	def sum_losses(self, track, frcnn):
		"""Calculate the losses.
		Args:
			track (list): Output from the MOT_Tracks dataloader
			frcnn (nn.Module): The faster rcnn network
		"""
		# check if using precalculated conv layer
		if 'conv' in track[0].keys():
			prec = True
		else:
			prec = False

		cl = 1 #Human

		self._losses = {}
		hidden = self.init_hidden()

		seq_len = len(track)

		# track begins at person gt
		pos = track[0]['gt'].cuda()
		pos = clip_boxes(Variable(pos), track[0]['im_info'][0][:2]).data

		bbox_losses = []

		for i in range(1,seq_len):
			t = track[i]
			# get fc7 in new image at old position
			if prec:
				frcnn._net_conv = Variable(t['conv'][0]).cuda()
				_, _, _, _ = frcnn.test_image(None, None, pos)
			else:
				_, _, _, _ = frcnn.test_image(t['data'][0], t['im_info'][0], pos)
		
			search_features = Variable(frcnn.get_fc7())

			bbox_reg, hidden = self.forward(search_features, hidden)
			
			# now regress with the output of the LSTM
			boxes = bbox_transform_inv(pos, bbox_reg.data)
			# seems like only can handle variables because hasattr(x,'data') throws error for tensors
			boxes = clip_boxes(Variable(boxes), t['im_info'][0][:2]).data
			pos = boxes

			# Calculate the regression targets
			target = t['gt'].cuda()

			bbox_targets = bbox_transform(pos, target)
			greater_than = torch.ge(bbox_targets, 0.5)
			if greater_than.sum() > 1:
				print("bbox targets: {}".format(bbox_targets))
			#bbox_targets = bbox_targets.clamp(min=-0.5, max=0.5)
			bbox_targets = Variable(bbox_targets)

			# Calculate the bbox losses
			loss = F.smooth_l1_loss(bbox_reg, bbox_targets)
			bbox_losses.append(loss)

		bbox_loss = torch.cat(bbox_losses).mean()

		self._losses['bbox'] = bbox_loss

		self._losses['total_loss'] = bbox_loss

		return self._losses