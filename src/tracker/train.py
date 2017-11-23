
#from model.config import cfg as frcnn_cfg

from .config import cfg, get_output_dir
from .datalayer import DataLayer
from .utils import plot_correlation, bbox_overlaps, plot_tracks
from .lstm_tracker import LSTMTracker


import torch
from torch.autograd import Variable
import numpy as np
import os
from os import path as osp
import glob
import time

import tensorboardX as tb




class Solver(object):

	def __init__(self, tfrcnn, rnn, db, db_val, output_dir, tb_dir):
		"""init

		Expects a list of 2 fully initialized tfrcnn in eval mode
		"""
		self.tfrcnn = tfrcnn
		self.rnn = rnn
		self.db = db
		self.db_val = db_val
		self.output_dir = output_dir
		self.tb_dir = tb_dir
		# Simply put '_val' at the end to save the summaries from the validation set
		self.tb_val_dir = tb_dir + '_val'
		if not os.path.exists(self.tb_val_dir):
			os.makedirs(self.tb_val_dir)
		self._losses = {}

	def construct_graph(self):
		# Set the random seed
		torch.manual_seed(cfg.RNG_SEED)
		

		lr = cfg.TRAIN.LEARNING_RATE

		params = []
		for key, value in dict(self.rnn.named_parameters()).items():
			if value.requires_grad:
				params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
				#if 'bias' in key:
				#	params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), 'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
				#else:
				#	params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

		self.optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

		# Write the train and validation information to tensorboard
		self.writer = tb.SummaryWriter(self.tb_dir)
		self.valwriter = tb.SummaryWriter(self.tb_val_dir)

	def init_weigts(self):
		pass

	def snapshot(self, iter):

		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)

		filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pth'
		filename = os.path.join(self.output_dir, filename)
		torch.save(self.rnn.state_dict(), filename)
		print('Wrote snapshot to: {:s}'.format(filename))

	def from_snapshot(self, sfile):
		print('Restoring model snapshots from {:s}'.format(sfile))
		self.rnn.load_state_dict(torch.load(str(sfile)))
		print('Restored.')

	def find_previous(self):
		sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pth')
		sfiles = glob.glob(sfiles)
		sfiles.sort(key=os.path.getmtime)



	def train_model(self, max_iters):
		self.data_layer = DataLayer(self.db.get_data())
		self.data_layer_val = DataLayer(self.db_val.get_data(), val=True)

		# Construct the computation graph
		self.construct_graph()

		iter = 1

		total_loss = 0
		cross_entropy = 0
		binary_cross_entropy = 0


		now = time.time()

		while iter < max_iters + 1:
			mb = self.data_layer.forward()
			blobs = mb['blobs']
			rois0, fc70, score0 = self.tfrcnn[0].test_image(blobs['data'][0], blobs['im_info'])
			rois1, fc71, score1 = self.tfrcnn[1].test_image(blobs['data'][1], blobs['im_info'])

			# score (300x2)
			rois0.volatile = False
			rois1.volatile = False
			fc70.volatile = False
			fc71.volatile = False
			score0.volatile = False
			score1.volatile = False


			losses = self.rnn.train_step(self.optimizer, rois0, rois1, fc70, fc71, score0, score1, mb)

			#plot_correlation(mb['im_paths'], mb['blobs'][0]['im_info'], mb['blobs'][1]['im_info'], cor, r[0],r[1])

			total_loss += losses['total_loss'].data.cpu().numpy()[0]
			cross_entropy += losses['cross_entropy'].data.cpu().numpy()[0]
			binary_cross_entropy += losses['binary_cross_entropy'].data.cpu().numpy()[0]

			if iter % 100 == 0:
				next_now = time.time()
				print("Iteration: {}, {}s/it".format(iter, (next_now-now)/100))
				now = next_now

				total_loss = total_loss/100
				cross_entropy = cross_entropy/100
				binary_cross_entropy = binary_cross_entropy/100

				print("Total Loss: {}".format(total_loss))
				print("Cross Entropy Loss: {}".format(cross_entropy))
				print("Binary Cross Entropy Loss: {}".format(binary_cross_entropy))

				self.writer.add_scalar("tota_loss", total_loss, iter)
				self.writer.add_scalar("cross_entropy", cross_entropy, iter)
				self.writer.add_scalar("binary_cross_entropy", binary_cross_entropy, iter)

				total_loss = 0
				cross_entropy = 0
				binary_cross_entropy = 0

			if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
				#self.snapshot(iter)
				last_snapshot_iter = iter

				# Print a picture of the current tracks
				tracks = self.rnn.test(rois0, rois1, fc70, fc71, score0, score1, mb)
				plot_tracks(mb, tracks, None, get_output_dir("track_plot_test1"), iter)


			iter += 1

		#if last_snapshot_iter != iter - 1:
		#	self.snapshot(iter - 1)

		self.writer.close()
		self.valwriter.close()



def train_net(tfrcnn, db, db_val, output_dir, tb_dir, max_iters=100000):
	# number of inputs of rnn at the moment 300x300 correlation matrix,
	# 2x300 person scores, 300x300 IoU = 180.600
	# NEW: input now downsampled to 300xN, so with N=20 now 6000
	rnn = LSTMTracker(6000, cfg.LSTM.HIDDEN_NUM, cfg.LSTM.LAYERS)
	rnn.train()
	rnn.cuda()
	solver = Solver(tfrcnn, rnn, db, db_val, output_dir, tb_dir)
	solver.train_model(max_iters)


