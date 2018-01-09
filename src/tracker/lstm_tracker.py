
from .utils import bbox_overlaps, hungarian_soft, plot_bb
from .correlation_layer import correlation_layer
from .config import cfg

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

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
		# 1805 is composed of 300 cor, 4*300+4 positions (1204) and 300+1 person scores
		self.input_lin = nn.Linear(601,cfg.LSTM.SAMPLE_N)
		self.input_dropout = nn.Dropout(p=0.5)

		self.hidden = self.init_hidden()

		#for name, param in self.lstm.named_parameters():
		#	print(name)
		#	print(param)

		self.init_weights()


	def init_weights(self):
		#print("here")
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

	def _generate_input(self, rois0, rois1, fc70, fc71, score0, score1, blobs, seq_len):
		"""Input generator

		Args:
			rois0/1: (300,5) region of interests matrices
			fc70/1: feature matrices for the rois
			score0/1: (300) vector of scores
			perm: LongTensor containing the permutation
		"""

		inputs = []
		inputs.append(self._generate_scores(score0, score1))
		#inputs.append(self._generate_correlation(fc70, fc71))
		inputs.append(self._generate_overlaps(rois0, rois1))
		#inputs.append(self._generate_positions(rois0, rois1, blobs))
		#inputs.append(self._generate_distances(rois0, rois1))

		# Now concatenate the all input matrices to form 300xn matrix
		# linear layer exepcts (N, in_features)
		feature_mat = torch.cat(inputs,1)
		input = self.input_lin(feature_mat).view(1,1,-1)
		#input = feature_mat.view(1,1,-1)
		
		#input = self.input_dropout(input)

		# concatenate input so that its to generate seq_len inputs
		if seq_len > 1:
			inp = []
			for i in range(seq_len):
				inp.append(input)
			input = torch.cat(inp)

		return input

	def _generate_distances(self, rois0, rois1):
		# should contain concatenated vectors of size 4 with
		# x_center,y_center,width,height
		pos0 = Variable(torch.zeros(300*4)).cuda()
		pos1 = Variable(torch.zeros(300*4)).cuda()

		#rois0 = rois0[:,1:]
		#rois1 = rois1[:,1:]

		# x_center
		pos0[0::4] = (rois0[:,0] + rois0[:,2]) / 2
		pos1[0::4] = (rois1[:,0] + rois1[:,2]) / 2

		# y_center
		pos0[1::4] = (rois0[:,1] + rois0[:,3]) / 2
		pos1[1::4] = (rois1[:,1] + rois1[:,3]) / 2

		# width
		pos0[2::4] = (rois0[:,2] - rois0[:,0])
		pos1[2::4] = (rois1[:,2] - rois1[:,0])

		# height
		pos0[3::4] = (rois0[:,3] - rois0[:,1])
		pos1[3::4] = (rois1[:,3] - rois1[:,1])

		# generate the matrix (300,902)
		res = []
		for i in range(300):
			dist = torch.sqrt(torch.pow(pos0[0::4]-pos1[i*4],2) + torch.pow(pos0[1::4]-pos1[i*4+1],2))
			res.append(torch.cat((dist,pos0[2::4],pos0[3::4],pos1[i*4+2:(i+1)*4])).view(1,-1))
		res = torch.cat(res)

		return res



	def _generate_scores(self, score0, score1):
		"""Generates the score matrix"""
		# only keep score for person, throw away background
		score0 = score0[:,1]
		score1 = score1[:,1]

		# Generate 300x301 score matrix, every line should contain 300 score1 entries + i score0
		sc = []
		for i in range(300):
			sc.append(torch.cat((score1, score0[i])).view(1,-1))
		sc = torch.cat(sc)

		return sc

	def _generate_correlation(self, fc70, fc71):
		cor = correlation_layer(fc70,fc71)

		return cor

	def _generate_overlaps(self, rois0, rois1):
		"""
		if perm0 is not None and perm1 is not None:
			rois0 = rois0[:,1:]
			rois1 = rois1[:,1:]
			ov = bbox_overlaps(rois0[perm0], rois1[perm1])
		else:
			ov = bbox_overlaps(rois0[:,1:], rois1[:,1:])
		"""
		ov = bbox_overlaps(rois0, rois1)

		return ov


	def _generate_positions(self, rois0, rois1, blobs):
		"""Generates the positions matrix"""

		# generate positional input
		im_info = Variable(torch.from_numpy(blobs['im_info'][0])).cuda()
		height = im_info[0]
		width = im_info[1]

		#print("height: {}, width:{}".format(height,width))

		# should contain concatenated vectors of size 4 with
		# x_center,y_center,width,height e [0,1]
		pos0 = Variable(torch.zeros(300*4)).cuda()
		pos1 = Variable(torch.zeros(300*4)).cuda()

		#rois0 = rois0[:,1:]
		#rois1 = rois1[:,1:]

		# x_center
		pos0[0::4] = (rois0[:,0] + rois0[:,2]) / 2 / width
		pos1[0::4] = (rois1[:,0] + rois1[:,2]) / 2 / width

		# y_center
		pos0[1::4] = (rois0[:,1] + rois0[:,3]) / 2 / height
		pos1[1::4] = (rois1[:,1] + rois1[:,3]) / 2 / height

		# width
		pos0[2::4] = (rois0[:,2] - rois0[:,0]) / width
		pos1[2::4] = (rois1[:,2] - rois1[:,0]) / width

		# height
		pos0[3::4] = (rois0[:,3] - rois0[:,1]) / height
		pos1[3::4] = (rois1[:,3] - rois1[:,1]) / height

		# generate the matrix (300,1204)
		pos = []
		for i in range(300):
			pos.append(torch.cat((pos0,pos1[i*4:(i+1)*4])).view(1,-1))
		pos = torch.cat(pos)

		return pos

	def test(self, rois0, rois1, fc70, fc71, score0, score1, blobs):
		self.hidden = self.init_hidden()
		self.eval()

		# generate input with seq_len = 1 as we don't know how many tracks there are
		input = self._generate_input(rois0, rois1, fc70, fc71, score0, score1, blobs, 1)

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


	def train_step(self, optimizer, rois0, rois1, fc70, fc71, score0, score1, blobs):
		# Expects optimizer, rois, fc7, and gt tracks

		
		# Step 1. Remember that Pytorch accumulates gradients.
		# We need to clear them out before each instance
		self.zero_grad()

		self.sum_losses(rois0, rois1, fc70, fc71, score0, score1, blobs)

		# update model
		self._losses['total_loss'].backward()
		optimizer.step()

		return self._losses

	def get_losses(self, rois0, rois1, fc70, fc71, score0, score1, blobs):
		"""Returns losses without training step, useful for validation set"""
		self.eval()
		self.sum_losses(rois0, rois1, fc70, fc71, score0, score1, blobs)
		self.train()

		return self._losses

	def sum_losses(self, rois0, rois1, fc70, fc71, score0, score1, blobs):
		# Clear losses
		self._losses = {}

		# Also, we need to clear out the hidden state of the LSTM,
		# detaching it from its history on the last instance.
		self.hidden = self.init_hidden()

		# This are the resized tracks
		tracks = blobs['tracks']

		# generate permutations if wanted
		if cfg.LSTM.PERM_INP:
			perm0 = torch.randperm(300).cuda()
			perm1 = torch.randperm(300).cuda()
		else:
			perm0 = torch.arange(300).cuda()
			perm1 = torch.arange(300).cuda()

		# permute all inputs
		fc70 = fc70[perm0]
		fc71 = fc71[perm1]
		rois0 = rois0[:,1:]
		rois1 = rois1[:,1:]
		rois0 = rois0[perm0]
		rois1 = rois1[perm1]

		# Assign anchor boxes to the gt tracks, this should be what the network predicts
		# tracks (i,1,4)
		gt_tracks = Variable(torch.from_numpy(tracks)).cuda()
		
		gt_ov0 = bbox_overlaps(gt_tracks[:,0,:], rois0)
		gt_ov1 = bbox_overlaps(gt_tracks[:,1,:], rois1)

		"""
		# only select gt where IoU is greater than 0.5
		gt_ov0 = torch.ge(gt_ov0, 0.5)
		gt_ov1 = torch.ge(gt_ov1, 0.5)

		# are there matching proposals? (in both gt_ov0 and gt_ov1)
		gt_exist = torch.nonzero(gt_ov0.sum(1).data*gt_ov1.sum(1).data).view(-1)

		# select only targets where there exist matching proposals
		gt_ov0 = gt_ov0[gt_exist]
		gt_ov1 = gt_ov1[gt_exist]
		
		# sample randomly from targets
		ind0, ind1 = self.random_targets(gt_ov0, gt_ov1)
		"""

		# take maximum IoU sample
		sc0, ind0 = torch.max(gt_ov0,1)
		sc1, ind1 = torch.max(gt_ov1,1)

		# get on 2 more to train when to stop
		#seq_len = tracks.shape[0] + 2
		seq_len = gt_ov0.size(0) + 2

		input = self._generate_input(rois0, rois1, fc70, fc71, score0, score1, blobs, seq_len)

		bbox0, bbox1, prob = self.forward(input)
		
		# Throw away the last two bb as they are not needed
		bbox0 = bbox0[:-2]
		bbox1 = bbox1[:-2]

		#plot_bb(mb, rois0[ind0.data][:,1:], rois1[ind1.data][:,1:], mb['tracks'])

		# We want now the best matchings between what out net predicted and what it should predict
		# NOT!!!! Apply softmax before
		#print(ind0)
		ind0_sort, ind1_sort = hungarian_soft(bbox0, bbox1, ind0, ind1)

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


	def random_targets(self, gt_ov0, gt_ov1):
		# sample randomly which sample to take
		ind0 = torch.LongTensor(gt_ov0.size(0)).cuda()
		ind1 = torch.LongTensor(gt_ov1.size(0)).cuda()
		for i in range(gt_ov0.size(0)):
			tmp0 = gt_ov0[i]
			tmp1 = gt_ov1[i]

			rand0 = np.random.randint(gt_ov0.sum(1)[i].data.cpu().numpy())
			rand1 = np.random.randint(gt_ov1.sum(1)[i].data.cpu().numpy())

			ind0[i] = torch.nonzero(tmp0.data)[rand0][0]
			ind1[i] = torch.nonzero(tmp1.data)[rand1][0]

		#sc0, indd0 = torch.max(gt_ov0,1)
		#print(torch.stack((indd0,ind0),1))

		#print(ind0)
		#print(indd0)

		return Variable(ind0), Variable(ind1)






