# LSTM based regressor, one instance for every person

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np

from model.bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes

from .appearance_lstm import Appearance_LSTM


class LSTM_Regressor(nn.Module):
	"""LSTM Regressor Class

	A regressor with a LSTM module with input 501. Inputs are compressed fc7 features to 500 with a FC layer and
	the person score outputted by the frcnn. The same FC layer is used to compress the fc7 features in the search
	region to 500. Then LSTM hidden layer and compressed search features are concatenated in a FC layer.
	Ouputs are the regression and a score if the track should be kept alive or not.
	"""
	def __init__(self, feature_size, appearance_lstm, sample_search_regions, sample_num, comp_features,
				 comp_features_in, search_region_factor, alive_mode):
		super(LSTM_Regressor, self).__init__()
		self.name = "LSTM_Regressor"

		assert alive_mode == 'vis' or alive_mode == 'kill', "[!] Invalid mode: {}".format(alive_mode)
		self.alive_mode = alive_mode

		self.search_region_factor = search_region_factor

		# sample more search regions based on gt to better train regression precision
		self.sample_search_regions = sample_search_regions
		self.sample_num = sample_num
		self.sampler = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))

		# Input compressed, regressed features in old image + person score from regressed box old image
		self.appearance_lstm = Appearance_LSTM(**appearance_lstm)
		# compression layer for features
		self.comp_features = comp_features
		if comp_features:
			self.compression = nn.Sequential(nn.Linear(comp_features_in, feature_size), nn.ReLU())

		self.regressor = nn.Sequential(nn.Linear(feature_size+self.appearance_lstm.output_dim, 501), nn.ReLU(),
			nn.Linear(501, 501), nn.ReLU(), nn.Linear(501, 5))

	def forward(self, old_features, old_scores, hidden, search_features, mode='TEST'):
		"""
		# Output of lstm (seq_len, batch, hidden_size)
		Args:
			input (Variable): input with size (seq_len, batch, input_size)
			hidden (Variable): hidden layer for the tracks (hidden, cell) each with dim (num_layers, batch, hidden_size)
		"""

		hidden = self.pred_lstm(old_features, old_scores, hidden, mode)

		bbox_reg, alive = self.pred_head(search_features, mode)

		return bbox_reg, alive, hidden

	def pred_lstm(self, old_features, old_scores, hidden, mode):
		if self.comp_features:
			old_features = self.compression(old_features)
		lstm_inp = torch.cat((old_features, old_scores),1).view(1,-1,self.appearance_lstm.input_dim)
		lstm_out, hidden = self.appearance_lstm(lstm_inp, hidden)

		self.lstm_out = lstm_out

		return hidden

	def pred_head(self, search_features, mode, mult_lstm_out=None):
		if self.comp_features:
			search_features = self.compression(search_features)
		lstm_out = self.lstm_out

		if mult_lstm_out:
			lstm_out = torch.cat([lstm_out for _ in range(mult_lstm_out)], 1)

		regressor_out = self.regressor(torch.cat((lstm_out[0,:,:],search_features),1))

		bbox_reg = regressor_out[:,:4]
		if mode=='TEST' or self.alive_mode=='vis':
			alive = F.sigmoid(regressor_out[:,4])
		else:
			alive = regressor_out[:,4]

		return bbox_reg, alive

	def sum_losses(self, track, frcnn, cnn=None):
		"""Calculates loss
		Args:
			track (list): Output from the MOT_Tracks dataloader, length at least 2
			frcnn (nn.Module): The faster rcnn network
			cnn (nn.Module): A alternative network for the feature extraction if frcnn should not be used
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

		# Check if first frame is a valid person
		assert track[0]['active'][0] == 1, "[!] First frame in track has to be active!"

		# track begins at person gt
		pos = track[0]['gt'].cuda()
		if prec:
			frcnn._net_conv = Variable(track[0]['conv'][0]).cuda()
			_, scores, _, _ = frcnn.test_image(None, None, pos)
		else:
			_, scores, _, _ = frcnn.test_image(track[0]['data'][0], track[0]['im_info'][0], pos)

		if cnn:
			old_features = cnn.test_image(track[0]['data'][0], pos)
		else:
			old_features = Variable(frcnn.get_fc7())
		old_score = Variable(scores[:1,cl].view(1,1))
		old_score = Variable(scores[:1,cl].view(1,1).fill_(0))

		bbox_losses = []
		alive_losses = []

		for i in range(1,seq_len):
			t = track[i]

			if self.sample_search_regions:
				gt = t['gt'][0].cuda()
				x0 = gt[0]
				y0 = gt[1]
				x1 = gt[2]
				y1 = gt[3]

				cx = (x1+x0)/2
				cy = (y1+y0)/2
				w = x1-x0+1
				h = y1-y0+1

				samples = []

				for _ in range(self.sample_num):
					new_cx = cx + w * 0.2*sample_limits(self.sampler,-2,2)
					new_cy = cy + h * 0.1*sample_limits(self.sampler,-2,2)
					new_w = w + w * 0.1*sample_limits(self.sampler, -2, 2)
					new_h = h + h * 0.1*sample_limits(self.sampler, -2, 2)

					new_x0 = new_cx - new_w/2
					new_x1 = new_cx + new_w/2
					new_y0 = new_cy - new_h/2
					new_y1 = new_cy + new_h/2

					s = torch.stack([new_x0, new_y0, new_x1, new_y1], 1).cuda()

					samples.append(s)

				samples = torch.cat(samples, 0)

				samples = clip_boxes(Variable(samples), t['im_info'][0][:2]).data

				# augument all variables
				pos = torch.cat([pos, samples], 0)

			# Increase search window size
			if self.search_region_factor != 1.0:
				pos = increase_search_region(pos, self.search_region_factor)
				pos = clip_boxes(Variable(pos), t['im_info'][0][:2]).data

			# get fc7 in new image at old position
			if prec:
				frcnn._net_conv = Variable(t['conv'][0]).cuda()
				_, _, _, _ = frcnn.test_image(None, None, pos)
			else:
				_, _, _, _ = frcnn.test_image(t['data'][0], t['im_info'][0], pos)
			
			if cnn:
				search_features = cnn.test_image(t['data'][0], pos)
			else:
				search_features = Variable(frcnn.get_fc7())

			if self.sample_search_regions:
				hidden = self.pred_lstm(old_features, old_score, hidden, 'TRAIN')
				bbox_reg, alive = self.pred_head(search_features, 'TRAIN', self.sample_num + 1)
			else:
				bbox_reg, alive, hidden = self.forward(old_features, old_score, hidden, search_features, 'TRAIN')
			
			# now regress with the output of the LSTM
			boxes = bbox_transform_inv(pos, bbox_reg.data)
			# seems like only can handle variables because hasattr(x,'data') throws error for tensors
			boxes = clip_boxes(Variable(boxes), t['im_info'][0][:2]).data
			pos = boxes

			# get the fc7 on the image on the regressed coordinates
			_, scores, _, _ = frcnn.test_rois(pos[:1])
			if cnn:
				old_features = cnn.test_rois(pos)
			else:
				old_features = Variable(frcnn.get_fc7())
			old_score = Variable(scores[:,cl].view(1,1))
			old_score = Variable(scores[:1,cl].view(1,1).fill_(0))

			if t['active'][0] == 1:
				# Calculate the regression targets
				target = t['gt'].cuda()
				if self.sample_search_regions:
					target = torch.cat([target for _ in range(self.sample_num+1)], 0)

				bbox_targets = Variable(bbox_transform(pos, target))

				# Calculate the bbox losses
				loss = F.smooth_l1_loss(bbox_reg, bbox_targets)
				bbox_losses.append(loss)

				alive_tar = Variable(torch.Tensor(1).fill_(1)).cuda()
			else:
				alive_tar = Variable(torch.Tensor(1).fill_(0)).cuda()

			if self.alive_mode == 'vis':
				alive_tar = Variable(t['vis'][0:1].float().cuda())
				loss = F.mse_loss(alive[:1].view(-1), alive_tar)
			else:
				loss = F.binary_cross_entropy_with_logits(alive[:1].view(-1).contiguous(), alive_tar)


			#if self.sample_search_regions:
			#	alive_tar = torch.cat([alive_tar for _ in range(self.sample_num + 1)], 0)

			# calculate the alive losses
			#loss = F.binary_cross_entropy_with_logits(alive[:1].view(-1).contiguous(), alive_tar)
			#loss = F.binary_cross_entropy_with_logits(alive, alive_tar)
			#loss = F.mse_loss(alive[:1].view(-1), alive_tar)
			#loss = F.mse_loss(alive, alive_tar)
			alive_losses.append(loss)

			# Important if sampling
			pos = pos[:1]

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

def sample_limits(sampler, lower_bound=None, upper_bound=None):
	sample = sampler.sample()
	if sample[0] < lower_bound:
		sample = sample.new([lower_bound])
	if sample[0] > upper_bound:
		sample = sample.new([upper_bound])
	return sample

def increase_search_region(pos, factor):
	x0 = pos[:,0]
	y0 = pos[:,1]
	x1 = pos[:,2]
	y1 = pos[:,3]

	cx = (x0+x1)/2
	cy = (y0+y1)/2
	w = x1-x0
	h = y1-y0

	w = w * factor
	h = h * factor

	new_x0 = cx - w/2
	new_x1 = cx + w/2
	new_y0 = cy - h/2
	new_y1 = cy + h/2

	pos = torch.stack((new_x0, new_y0, new_x1, new_y1), 1)

	return pos

