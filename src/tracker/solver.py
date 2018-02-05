from random import shuffle
import numpy as np
import os

import torch
from torch.autograd import Variable

from .datalayer import DataLayer
from .config import cfg
from .utils import plot_tracks

import tensorboardX as tb


class Solver(object):
	default_sgd_args = {"lr": 1e-4,
						 "weight_decay": 0.0,
						 "momentum":0}

	def __init__(self, output_dir, tb_dir, optim=torch.optim.SGD, optim_args={}):

		optim_args_merged = self.default_sgd_args.copy()
		optim_args_merged.update(optim_args)
		self.optim_args = optim_args_merged
		self.optim = optim

		self.output_dir = output_dir
		self.tb_dir = tb_dir
		# Simply put '_val' at the end to save the summaries from the validation set
		self.tb_val_dir = tb_dir + '_val'
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)
		if not os.path.exists(self.tb_dir):
			os.makedirs(self.tb_dir)
		if not os.path.exists(self.tb_val_dir):
			os.makedirs(self.tb_val_dir)

		# Set the seeds
		seed = cfg.RNG_SEED
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		np.random.seed(seed)
		torch.backends.cudnn.deterministic = True

		self._reset_histories()

	def _reset_histories(self):
		"""
		Resets train and val histories for the accuracy and the loss.
		"""
		self._losses = {}
		self._val_losses = {}

	def snapshot(self, model, iter):
		filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pth'
		filename = os.path.join(self.output_dir, filename)
		torch.save(model.state_dict(), filename)
		print('Wrote snapshot to: {:s}'.format(filename))

	def train(self, model, db_train, db_val=None, num_epochs=10, log_nth=0):
		"""
		Train a given model with the provided data.

		Inputs:
		- model: model object initialized from a torch.nn.Module
		- train_loader: train data in torch.utils.data.DataLoader
		- val_loader: val data in torch.utils.data.DataLoader
		- num_epochs: total number of training epochs
		- log_nth: log training accuracy and loss every nth iteration
		"""

		self.writer = tb.SummaryWriter(self.tb_dir)
		self.val_writer = tb.SummaryWriter(self.tb_val_dir)

		# filter out frcnn if this is added to the module
		parameters = [param for name, param in model.named_parameters() if 'frcnn' not in name]
		optim = self.optim(parameters, **self.optim_args)

		self._reset_histories()
		iter_per_epoch = db_train.size

		data_layer = DataLayer(db_train.get_data())
		if db_val:
			data_layer_val = DataLayer(db_val.get_data(), val=True)

		print('START TRAIN.')
		############################################################################
		# TODO:                                                                    #
		# Write your own personal training method for our solver. In Each epoch    #
		# iter_per_epoch shuffled training batches are processed. The loss for     #
		# each batch is stored in self.train_loss_history. Every log_nth iteration #
		# the loss is logged. After one epoch the training accuracy of the last    #
		# mini batch is logged and stored in self.train_acc_history.               #
		# We validate at the end of each epoch, log the result and store the       #
		# accuracy of the entire validation set in self.val_acc_history.           #
		#
		# Your logging should like something like:                                 #
		#   ...                                                                    #
		#   [Iteration 700/4800] TRAIN loss: 1.452                                 #
		#   [Iteration 800/4800] TRAIN loss: 1.409                                 #
		#   [Iteration 900/4800] TRAIN loss: 1.374                                 #
		#   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                                #
		#   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                                #
		#   ...                                                                    #
		############################################################################

		for epoch in range(num_epochs):
			# TRAINING

			for i in range(1,iter_per_epoch+1):
				#inputs, labels = Variable(batch[0]), Variable(batch[1])
				blobs = data_layer.forward()

				optim.zero_grad()
				losses = model.sum_losses(blobs)
				losses['total_loss'].backward()
				optim.step()

				for k,v in losses.items():
					if k not in self._losses.keys():
						self._losses[k] = []
					self._losses[k].append(v.data.cpu().numpy())

				if log_nth and i % log_nth == 0:
					print('[Iteration %d/%d]' % (i + epoch * iter_per_epoch,
																  iter_per_epoch * num_epochs))
					for k,v in self._losses.items():
						last_log_nth_losses = self._losses[k][-log_nth:]
						train_loss = np.mean(last_log_nth_losses)
						print('%s: %.3f' % (k, train_loss))
						self.writer.add_scalar(k, train_loss, i + epoch * iter_per_epoch)
						
	
			# VALIDATION
			if db_val and log_nth:
				model.eval()
				for i in range(log_nth):
					blobs = data_layer_val.forward()

					losses = model.sum_losses(blobs)

					for k,v in losses.items():
						if k not in self._val_losses.keys():
							self._val_losses[k] = []
						self._val_losses[k].append(v.data.cpu().numpy())
					
				model.train()
				for k,v in self._losses.items():
					last_log_nth_losses = self._val_losses[k][-log_nth:]
					val_loss = np.mean(last_log_nth_losses)
					self.val_writer.add_scalar(k, val_loss, (epoch+1) * iter_per_epoch)

				blobs_val = data_layer_val.forward()
				tracks_val = model.val_predict(blobs_val)
				im = plot_tracks(blobs_val, tracks_val)
				self.val_writer.add_image('val_tracks', im, (epoch+1) * iter_per_epoch)

			self._reset_histories()

		self.writer.close()
		self.val_writer.close()

		self.snapshot(model, num_epochs*iter_per_epoch)

		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################
		print('FINISH.')
