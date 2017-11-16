
from .utils import bbox_overlaps, hungarian
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

        self.hidden = self.init_hidden()

        self.cel = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()


	def init_hidden(self):
		# Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)))

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
        

    def train_step(self, optimizer, rois0, rois1, fc70, fc71, score0, score1, tracks)
    	# Expects optimizer, rois, fc7, and gt tracks


    	# Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        self.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        self.hidden = model.init_hidden()

        # Flatten out input
        cor = correlation_layer(fc70,fc71).view(-1)
        ov = bbox_overlaps(rois0[:,1:], rois1[:,1:]).view(-1)
        score0 = score0.view(-1)
        score1 = score1.view(-1)
        input = torch.cat(cor,ov,score0,score1).view(1,1,-1)

        # get on 2 more to train when to stop
        seq_len = tracks.shape[0] + 2

        # Build input of size seq_len by concatenating
        x = []
        for i in range(seq_len):
            x.append(input)

        input = torch.cat(x)

        bbox0, bbox1, prob = self.forward(input)

        # Throw away the last two bb as they are not needed
        bbox0 = bbox0[:-2]
        bbox1 = bbox1[:-2]

        # Get the index of the maximum 
        _, bbox0max = torch.max(bbox0, 1)
        _, bbox1max = torch.max(bbox1, 1)

        # Construct tracks variable (N,1,4)
        tracks = torch.cat((rois0[bbox0max,1:].view(-1,1,4),rois1[bbox1max,1:].view(-1,1,4)),1)

        # Construct gt tracks and probs variable
        gt_tracks = Variable(torch.from_numpy(tracks))
        gt_probs = Variable((torch.ones(seq_len-2),torch.zeros(2))).view(-1,1)

        # Calculate the hungarian matching and cost matrix
        row_ind, col_ind, track_IoU = hungarian(tracks, gt_tracks)

        # select only tracks where IoU is over 1
        inds = torch.ge(track_IoU[row_ind,col_ind], Variable(torch.ones(1)))

        # only keep the row_ind that fulfill the condition
        row_ind = row_ind[inds]

        # Make the indexes that are correctly predicted even better
        loss = self.cel(bbox0[row_inds], bbox0max[row_inds]) + self.cel(bbox1[row_inds], bbox1max[row_inds])

        # Add to the loss that the probability should stop at the end
        loss += self.bce(prob, gt_probs)

        # update model
        loss.backward()
        optimizer.step()


