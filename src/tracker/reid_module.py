import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np

from .appearance_lstm import Appearance_LSTM

class ReID_Module(nn.Module):

	def __init__(self, cnn, appearance_lstm, mode):
		super(ReID_Module, self).__init__()
		self.name = "ReID_Module"

		assert mode in ["TRAIN", "TEST"], "[!] Invalid mode: {}".format(mode)
		self.mode = mode

		self.appearance_lstm = Appearance_LSTM(**appearance_lstm)

		self.cnn = cnn

		self.fc = nn.Sequential(nn.Linear(cnn.output_dim+self.appearance_lstm.output_dim, cnn.output_dim), nn.ReLU(),
			nn.Linear(cnn.output_dim, 1))

	
	def test_rois(self, image, rois, lstm_out):
		feat =  self.cnn.test_rois(image, rois)
		if feat.size(0) != lstm_out.size(0):
			lstm_out = torch.cat([lstm_out for _ in range(int(feat.size(0)/lstm_out.size(0)))], 0)
		fc_out = self.pred_head(feat, lstm_out)
		return fc_out.detach()

	def feed_rois(self, image, rois, hidden):
		feat =  self.cnn.test_rois(image, rois)
		lstm_out, (h0, h1) = self.pred_lstm(feat, hidden)
		return lstm_out.detach(), (h0.detach(), h1.detach())

	def forward(self, features, hidden, test_features):
		pass

	def pred_lstm(self, features, hidden):
		lstm_inp = features.view(1,-1,self.appearance_lstm.input_dim)
		lstm_out, hidden = self.appearance_lstm(lstm_inp, hidden)

		return lstm_out[0,:,:], hidden

	def pred_head(self, test_features, lstm_out):
		fc_out = self.fc(torch.cat((lstm_out,test_features),1))

		if self.mode == "TRAIN":
			return fc_out
		elif self.mode == "TEST":
			return F.sigmoid(fc_out)
		else:
			raise NotImplementedError("Mode: {}".format(self.mode))


	def sum_losses(self, batch, lstm_seq_length):
		"""Calculate the losses.

		A batch consists of P different persons with K crops each.
		"""
		data = batch[0][0].cuda()

		embeddings = self.cnn.test_images(data)

		labels = batch[1][0]
		unique_labels = torch.from_numpy(np.unique(labels.numpy())).cuda()
		P = unique_labels.size(0)
		labels = labels.cuda()

		K =  torch.eq(labels, unique_labels[0]).sum()
		# for every label construct a tensor with K positive (lstm sequence and for testing) and K - lstm_seq_length negative samples
		res = []
		for l in unique_labels:
			eq = torch.eq(labels, l).nonzero().view(-1)
			ne = torch.ne(labels, l).nonzero().view(-1)
			ne = ne[torch.randperm(ne.size(0))[:K-lstm_seq_length].cuda()]
			res.append(torch.cat((eq,ne)))
		res = torch.stack(res, 0).long()

		hidden = self.init_hidden(P)

		lstm_out_list = []
		for i in range(lstm_seq_length):
			e = embeddings[res[:,i]]
			lstm_out, hidden = self.pred_lstm(e, hidden)
			lstm_out_list.append(lstm_out)

		out = []
		tar = []
		for i in range(lstm_seq_length, K):
			e = embeddings[res[:,i]]
			for lstm_out in lstm_out_list:
				o = self.pred_head(e, lstm_out)
				out.append(o)
				tar.append(torch.ones(o.size()).cuda())
		for i in range(K, 2*K - lstm_seq_length):
			e = embeddings[res[:,i]]
			for lstm_out in lstm_out_list:
				o = self.pred_head(e, lstm_out)
				out.append(o)
				tar.append(torch.zeros(o.size()).cuda())
		out = torch.cat(out, 1)
		tar = Variable(torch.cat(tar, 1))

		losses = {}
		losses['total_loss'] = F.binary_cross_entropy_with_logits(out, tar)
		
		return losses


	def init_hidden(self, minibatch_size=1):
		return self.appearance_lstm.init_hidden(minibatch_size)