from random import shuffle
import numpy as np
import os
import time
import fnmatch

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR

from ..utils import plot_tracks

import tensorboardX as tb


class Solver(object):
	default_sgd_args = {"lr": 1e-4,
						 "weight_decay": 0.0,
						 "momentum":0}
	default_optim_args = {"lr": 1e-4}

	def __init__(self, output_dir, tb_dir, optim='SGD', optim_args={}, lr_scheduler_lambda=None):

		optim_args_merged = self.default_optim_args.copy()
		optim_args_merged.update(optim_args)
		self.optim_args = optim_args_merged
		if optim == 'SGD':
			self.optim = torch.optim.SGD
		elif optim == 'Adam':
			self.optim = torch.optim.Adam
		else:
			assert False, "[!] No valid optimizer: {}".format(optim)

		self.lr_scheduler_lambda = lr_scheduler_lambda

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

		self._reset_histories()

	def _reset_histories(self):
		"""
		Resets train and val histories for the accuracy and the loss.
		"""
		self._losses = {}
		self._val_losses = {}

	def snapshot(self, model, iter):
		filename = model.name + '_iter_{:d}'.format(iter) + '.pth'
		filename = os.path.join(self.output_dir, filename)
		torch.save(model.state_dict(), filename)
		print('Wrote snapshot to: {:s}'.format(filename))

		# Delete old snapshots (keep minimum 3 latest)
		snapshots_iters = []

		onlyfiles = [f for f in os.listdir(self.output_dir) if os.path.isfile(os.path.join(self.output_dir, f))]

		for f in onlyfiles:
			if fnmatch.fnmatch(f, 'ResNet_iters_*.pth'):
				snapshots_iters.append(int(f.split('_')[2][:-4]))

		snapshots_iters.sort()

		for i in range(len(snapshots_iters) - 3):
			filename = model.name + '_iter_{:d}'.format(snapshots_iters[i]) + '.pth'
			filename = os.path.join(self.output_dir, filename)
			os.remove(filename)

	def train(self, model, train_loader, val_loader=None, num_epochs=10, log_nth=0, model_args={}):
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

		if self.lr_scheduler_lambda:
			scheduler = LambdaLR(optim, lr_lambda=self.lr_scheduler_lambda)
		else:
			scheduler = None

		self._reset_histories()
		iter_per_epoch = len(train_loader)

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
			if scheduler and epoch:
				scheduler.step()
				print("[*] New learning rate(s): {}".format(scheduler.get_lr()))

			now = time.time()

			for i, batch in enumerate(train_loader, 1):
				#inputs, labels = Variable(batch[0]), Variable(batch[1])
				

				optim.zero_grad()
				losses = model.sum_losses(batch, **model_args)
				losses['total_loss'].backward()
				optim.step()

				for k,v in losses.items():
					if k not in self._losses.keys():
						self._losses[k] = []
					self._losses[k].append(v.data.cpu().numpy())

				if log_nth and i % log_nth == 0:
					next_now = time.time()
					print('[Iteration %d/%d] %.3f s/it' % (i + epoch * iter_per_epoch,
																  iter_per_epoch * num_epochs, (next_now-now)/log_nth))
					now = next_now

					for k,v in self._losses.items():
						last_log_nth_losses = self._losses[k][-log_nth:]
						train_loss = np.mean(last_log_nth_losses)
						print('%s: %.3f' % (k, train_loss))
						self.writer.add_scalar(k, train_loss, i + epoch * iter_per_epoch)
						
	
			# VALIDATION
			if val_loader and log_nth:
				model.eval()
				for i, batch in enumerate(val_loader):

					losses = model.sum_losses(batch, **model_args)

					for k,v in losses.items():
						if k not in self._val_losses.keys():
							self._val_losses[k] = []
						self._val_losses[k].append(v.data.cpu().numpy())

					if i >= log_nth:
						break
					
				model.train()
				for k,v in self._losses.items():
					last_log_nth_losses = self._val_losses[k][-log_nth:]
					val_loss = np.mean(last_log_nth_losses)
					self.val_writer.add_scalar(k, val_loss, (epoch+1) * iter_per_epoch)

				#blobs_val = data_layer_val.forward()
				#tracks_val = model.val_predict(blobs_val)
				#im = plot_tracks(blobs_val, tracks_val)
				#self.val_writer.add_image('val_tracks', im, (epoch+1) * iter_per_epoch)

			self.snapshot(model, (epoch+1)*iter_per_epoch)

			self._reset_histories()

		self.writer.close()
		self.val_writer.close()

		############################################################################
		#                             END OF YOUR CODE                             #
		############################################################################
		print('FINISH.')
