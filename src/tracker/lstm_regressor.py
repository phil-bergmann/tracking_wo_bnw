# LSTM based regressor, one instance for every person

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model.bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes


class LSTM_Regressor(nn.Module):
	"""LSTM Regressor Class

	A regressor with a LSTM module with input 501. Inputs are compressed fc7 features to 500 with a FC layer and
	the person score outputted by the frcnn. The same FC layer is used to compress the fc7 features in the search
	region to 500. Then LSTM hidden layer and compressed search features are concatenated in a FC layer.
	Ouputs are the regression and a score if the track should be kept alive or not.
	"""
	def __init__(self, feature_size, appearance_lstm):
		super(LSTM_Regressor, self).__init__()
		self.name = "LSTM_Regressor"

		# Input compressed, regressed fc7 in old image + person score from regressed box old image
		self.appearance_lstm = appearance_lstm
		# compression layer for fc7 features
		self.compression = nn.Linear(4096, feature_size)

		self.regressor = nn.Sequential(nn.Linear(feature_size+self.appearance_lstm.output_dim, 501),
			nn.Linear(501, 501), nn.Linear(501, 5))

	def forward(self, old_fc7, scores, hidden, search_fc7):
		"""
		# Output of lstm (seq_len, batch, hidden_size)
		Args:
			input (Variable): input with size (seq_len, batch, input_size)
			hidden (Variable): hidden layer for the tracks (hidden, cell) each with dim (num_layers, batch, hidden_size)
		"""
		old_fc7_compressed = self.compression(old_fc7)
		lstm_inp = torch.cat((old_fc7_compressed, scores),1).view(1,-1,self.appearance_lstm.input_dim)
		lstm_out, hidden = self.appearance_lstm(lstm_inp, hidden)

		search_fc7_compressed = self.compression(search_fc7)
		regressor_out = self.regressor(torch.cat((lstm_out[0,:,:],search_fc7_compressed),1))

		bbox_reg = regressor_out[:,:4]
		alive = regressor_out[:,4]

		return bbox_reg, alive, hidden

	def sum_losses(self, track, frcnn):
		"""Calculates loss
		Args:
			track (list): Output from the MOT_Tracks dataloader, length at least 2
		"""

		# check if using precalculated conv layer
		if 'conv' in track[0].keys():
			prec = True
		else:
			prec = false

		cl = 1 #Human

		self._losses = {}
		hidden = self.init_hidden()

		seq_len = len(track)

		# Check if first frame is a valid person
		assert track[0]['active'][0] == 1, "[!] First frame in track has to be active!"

		# track begins at person gt
		pos = track[0]['gt'].cuda()
		if prec:
			frcnn._net_conv = Variable(track[0]['conv'][0]).cuda()
			_, _, _, _ = frcnn.test_image(None, None, pos)
		else:
			_, _, _, _ = frcnn.test_image(track[0]['data'][0], track[0]['im_info'][0], pos)
		old_fc7 = Variable(frcnn.get_fc7())

		bbox_losses = []
		alive_losses = []

		for i in range(1,seq_len):
			t = track[i]

			# get fc7 in new image at old position
			if prec:
				frcnn._net_conv = Variable(t['conv'][0]).cuda()
				_, scores, _, _ = frcnn.test_image(None, None, pos)
			else:
				_, scores, _, _ = frcnn.test_image(t['data'][0], t['im_info'][0], pos)
			score = Variable(scores[:,cl].view(1,1))
			search_fc7 = Variable(frcnn.get_fc7())

			bbox_reg, alive, hidden = self.forward(old_fc7, score, hidden, search_fc7)
			
			# now regress with the output of the LSTM
			boxes = bbox_transform_inv(pos, bbox_reg.data)
			# seems like only can handle variables because hasattr(x,'data') throws error for tensors
			boxes = clip_boxes(Variable(boxes), t['im_info'][0][:2]).data
			pos = boxes

			# get the fc7 on the image on the regressed coordinates
			_, _, _, _ = frcnn.test_rois(pos)
			old_fc7 = Variable(frcnn.get_fc7())

			if t['active'][0] == 1:
				# Calculate the regression targets
				bbox_targets = Variable(bbox_transform(pos, t['gt'].cuda()))

				# Calculate the bbox losses
				loss = F.smooth_l1_loss(bbox_reg, bbox_targets)
				bbox_losses.append(loss)

				alive_tar = Variable(torch.cuda.FloatTensor(1).fill_(1))
			else:
				alive_tar = Variable(torch.cuda.FloatTensor(1).fill_(0))

			# calculate the alive losses
			loss = F.binary_cross_entropy_with_logits(alive.view(-1), alive_tar)
			alive_losses.append(loss)

		if len(bbox_losses) > 0:
			bbox_loss = torch.cat(bbox_losses).mean()
		else:
			bbox_loss = Variable(torch.zeros(1)).cuda()
		alive_loss = torch.cat(alive_losses).mean()

		self._losses['bbox'] = bbox_loss
		self._losses['alive'] = alive_loss

		self._losses['total_loss'] = bbox_loss + alive_loss

		return self._losses

	def init_hidden(self, minibatch_size=1):
		return self.appearance_lstm.init_hidden(minibatch_size)

	#def load_state_dict(self, state_dict):
	#	"""Load everything but frcnn"""
	#
	#	pretrained_state_dict = {k: v for k,v in state_dict.items() if 'frcnn' not in k}
	#	updated_state_dict = self.state_dict()
	#	updated_state_dict.update(pretrained_state_dict)
	#	nn.Module.load_state_dict(self, updated_state_dict)
	#
	#def state_dict(self):
	#	"""Overwrite to ignore FRCNN part"""
	#	#state_dict = {k: v for k,v in self.state_dict().items() if 'frcnn' not in k}
	#	state_dict = {}
	#	print(self.state_dict())
	#	print(self.named_parameters())
	#	for name,param in self.named_parameters():
	#		print(name)
	#	return self.state_dict()