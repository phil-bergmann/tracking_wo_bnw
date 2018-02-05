
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Appearance_RNN(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
		super(Appearance_RNN, self).__init__()

		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.num_layers = num_layers

		self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)

		self.out = nn.Linear(hidden_dim, output_dim)

		self.hidden = self.init_hidden()

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
		out = self.out(lstm_out.view(-1, self.hidden_dim))

		return out