
from .utils import bbox_overlaps, hungarian_soft, plot_bb
from .correlation_layer import correlation_layer

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class LSTMTracker(nn.Module):

	def __init__(self, input_dim, hidden_dim, num_layers):
		super(LSTMTracker, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers

		# The LSTM takes word embeddings as inputs, and outputs hidden states
		# with dimensionality hidden_dim.
		self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)

		# The linear layer that maps from hidden state space to bbox indexes
		self.hidden2bbox0 = nn.Linear(hidden_dim, 300)
		self.hidden2bbox1 = nn.Linear(hidden_dim, 300)
		self.hidden2prob = nn.Linear(hidden_dim, 1)

		# Only with 2D Tensors, applies along dimension 1 (N,features)
		self.softmax = nn.Softmax()
		self.sigmoid = nn.Sigmoid()
		self.cel = nn.CrossEntropyLoss()
		self.bcell = nn.BCEWithLogitsLoss()

		# Layer for feature compression, from (300,901) to (300,N)
		# 901 is composed of 300 correlation features, 300 IoU values and 300+1 person scores
		self.input_lin = nn.Linear(901,20)

		self.hidden = self.init_hidden()

		#for name, param in self.lstm.named_parameters():
		#	print(name)
		#	print(param)


	def init_hidden(self):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)).cuda(),
				Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)).cuda())

	def forward(self, input):
		# Expects the input to be (seq_len, batch, input_size)
		# Output of lstm (seq_len, batch, hidden_size)
		lstm_out, self.hidden = self.lstm(input, self.hidden)

		# calculate the outputs, for easier processing throw away the batch dimension,
		# as it is 1 here
		bbox0 = self.hidden2bbox0(lstm_out.view(-1, self.hidden_dim))
		bbox1 = self.hidden2bbox1(lstm_out.view(-1, self.hidden_dim))
		prob = self.hidden2prob(lstm_out.view(-1, self.hidden_dim))

		return bbox0, bbox1, prob

	def _generate_input(self, cor, ov, score0, score1, seq_len):
		"""Generates the downsampled input."""

		# Generate 300x301 score matrix, every line should contain 300 score1 entries + i score0
		sc = []
		for i in range(300):
			sc.append(torch.cat((score1, score0[i])).view(1,-1))
		sc = torch.cat(sc)

		# Now concatenate the cor, ov and sc matrices to form the 300x901 matrix
		# linear layer exepcts (N, in_features)
		feature_mat = torch.cat((cor,ov,sc),1)
		input = self.input_lin(feature_mat).view(1,1,-1)

		# concatenate input so that its to generate seq_len inputs
		inp = []
		for i in range(seq_len):
			inp.append(input)
		input = torch.cat(inp)

		return input

	def test(self, rois0, rois1, fc70, fc71, score0, score1, mb):
		blobs = mb['blobs']
		self.hidden = self.init_hidden()
		self.eval()

		# Flatten out input
		cor = correlation_layer(fc70,fc71)
		ov = bbox_overlaps(rois0[:,1:], rois1[:,1:])
		# only keep score for person, throw away background
		score0 = score0[:,1]
		score1 = score1[:,1]

		# generate input with seq_len = 1 as we don't know how many tracks there are
		input = self._generate_input(cor, ov, score0, score1, 1)

		tracks = []

		# constrain max tracks to 50 for now to avoid endless loop
		for i in range(50):
			bbox0, bbox1, prob = self.forward(input)
			# apply sigmoid to prob to get result
			prob = self.sigmoid(prob)[0,0].data.cpu().numpy()
			if prob < 0.5:
				break

			# Get the index of the maximum, no softmax needed as max still max after softmax
			_, bbox0max = torch.max(bbox0, 1)
			_, bbox1max = torch.max(bbox1, 1)


			t0 = rois0[bbox0max.data,:]
			t1 = rois1[bbox1max.data,:]
			
			t = torch.cat((t0[:,1:],t1[:,1:]))

			tracks.append(t)

		if tracks:
			tracks = torch.stack(tracks)
		else:
			tracks = Variable(torch.Tensor(0,2,4))

		self.train()

		return tracks


	def train_step(self, optimizer, rois0, rois1, fc70, fc71, score0, score1, mb):
		# Expects optimizer, rois, fc7, and gt tracks
		blobs = mb['blobs']
		# This is the resized tracks
		tracks = blobs['tracks']

		# Clear losses
		self._losses = {}

		# Step 1. Remember that Pytorch accumulates gradients.
		# We need to clear them out before each instance
		self.zero_grad()

		# Also, we need to clear out the hidden state of the LSTM,
		# detaching it from its history on the last instance.
		self.hidden = self.init_hidden()


		# Flatten out input
		cor = correlation_layer(fc70,fc71)
		ov = bbox_overlaps(rois0[:,1:], rois1[:,1:])
		# only keep score for person, throw away background
		score0 = score0[:,1]
		score1 = score1[:,1]

		# get on 2 more to train when to stop
		seq_len = tracks.shape[0] + 2

		input = self._generate_input(cor, ov, score0, score1, seq_len)

		bbox0, bbox1, prob = self.forward(input)
		
		# Throw away the last two bb as they are not needed
		bbox0 = bbox0[:-2]
		bbox1 = bbox1[:-2]

		# Assign anchor boxes to the gt tracks, this should be what the network predicts
		# tracks (i,1,4)
		gt_tracks = Variable(torch.from_numpy(tracks)).cuda()
		anchor_gt_ov0 = bbox_overlaps(gt_tracks[:,0,:], rois0[:,1:])
		anchor_gt_ov1 = bbox_overlaps(gt_tracks[:,1,:], rois1[:,1:])

		sc0, ind0 = torch.max(anchor_gt_ov0,1)
		sc1, ind1 = torch.max(anchor_gt_ov1,1)

		#plot_bb(mb, rois0[ind0.data][:,1:], rois1[ind1.data][:,1:], mb['tracks'])

		# We want now the best matchings between what out net predicted and what it should predict
		# Apply softmax before
		ind0_sort, ind1_sort = hungarian_soft(self.softmax(bbox0), self.softmax(bbox1), ind0, ind1)

		# Build gt probabilities [1,1,1,...,1,0,0]
		gt_probs = Variable(torch.cat((torch.ones(seq_len-2),torch.zeros(2))).view(-1,1)).cuda()

		# Begin to construct the loss
		cross_entropy = self.cel(bbox0, ind0_sort) + self.cel(bbox1, ind1_sort)

		# Add to the loss that the probability should stop at the end
		binary_cross_entropy = self.bcell(prob, gt_probs)

		total_loss = cross_entropy + binary_cross_entropy

		self._losses['cross_entropy'] = cross_entropy
		self._losses['binary_cross_entropy'] = binary_cross_entropy
		self._losses['total_loss'] = total_loss

		# update model
		total_loss.backward()
		optimizer.step()

		return self._losses


