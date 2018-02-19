# LSTM based regressor, one instance for every person

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model.bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes


class LSTM_Regressor(nn.Module):
	"""LSTM Regressor Class

	Inputs for the LSTM should be features of the regressed person in image t-1 so that the
	LSTM can learn how the person looks like, features at position t-1 in image at t so the
	network can regress to the new position and person score output of the FRCNN so the net-
	work can decide when the track is not a person anymore.
	Ouputs are the regression and a score if the track should be kept alive or not.
	"""
	def __init__(self, hidden_dim, num_layers=1, input_dim=4096*2):
		super(LSTM_Regressor, self).__init__()
		self.name = "LSTM_Regressor"
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.input_dim = input_dim

		# Input at the moment regressed fc7 from old image and search region in new image 
		self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)

		#self.frcnn = frcnn

		# The linear layers that map from hidden state space to output
		self.regress = nn.Linear(hidden_dim, 4)
		# The output that tells if we want to keep the track alive or let it die
		self.alive = nn.Linear(hidden_dim, 1)

		self.init_weights()

	def forward(self, input, hidden):
		"""
		# Output of lstm (seq_len, batch, hidden_size)
		Args:
			input (Variable): input with size (seq_len, batch, input_size)
			hidden (Variable): hidden layer for the tracks (hidden, cell) each with dim (num_layers, batch, hidden_size)
		"""
		lstm_out, hidden = self.lstm(input, hidden)

		bbox_reg = self.regress(lstm_out.view(-1, self.hidden_dim))
		alive = self.alive(lstm_out.view(-1, self.hidden_dim))

		return bbox_reg, alive, hidden

	def sum_losses(self, track, frcnn):
		"""Calculates loss
		Args:
			track (list): Output from the MOT_Tracks dataloader, length at least 2
		"""

		cl = 1 #Human

		self._losses = {}
		hidden = self.init_hidden()

		seq_len = len(track)

		# Check if first frame is a valid person
		assert track[0]['active'][0] == 1, "[!] First frame in track has to be active!"

		# track begins at person gt
		pos = track[0]['gt'].cuda()
		_, _, _, _ = frcnn.test_image(track[0]['data'][0], track[0]['im_info'][0], pos)
		old_fc7 = frcnn.get_fc7()

		bbox_losses = []
		alive_losses = []

		for i in range(1,seq_len):
			t = track[i]

			# get fc7 in new image at old position
			_, _, _, _ = frcnn.test_image(t['data'][0], t['im_info'][0], pos)
			new_fc7 = frcnn.get_fc7()

			input = Variable(torch.cat((old_fc7, new_fc7), 1).view(1,1,-1))

			bbox_reg, alive, hidden = self.forward(input, hidden)

			
			# now regress with the output of the LSTM
			boxes = bbox_transform_inv(pos, bbox_reg.data)
			# seems like only can handle variables because hasattr(x,'data') throws error for tensors
			boxes = clip_boxes(Variable(boxes), t['im_info'][0][:2]).data
			pos = boxes

			# get the fc7 on the image on the regressed coordinates
			_, _, _, _ = frcnn.test_rois(pos)
			old_fc7 = frcnn.get_fc7()

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

	def init_weights(self):
		#for name, param in self.lstm.named_parameters():
		#	print(name)
		#	print(param.size())

		# initialize forget gates to 1
		for names in self.lstm._all_weights:
			for name in filter(lambda n: "bias" in n,  names):
				bias = getattr(self.lstm, name)
				n = bias.size(0)
				start, end = n//4, n//2
				bias.data[start:end].fill_(1.)

	def init_hidden(self, minibatch_size=1):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (Variable(torch.zeros(self.num_layers, minibatch_size, self.hidden_dim)).cuda(),
				Variable(torch.zeros(self.num_layers, minibatch_size, self.hidden_dim)).cuda())

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