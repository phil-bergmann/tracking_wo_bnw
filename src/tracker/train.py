
#from model.config import cfg as frcnn_cfg

from .config import cfg
from .datalayer import DataLayer
from .utils import plot_correlation, bbox_overlaps
from .lstm_tracker import LSTMTracker


import torch
from torch.autograd import Variable
import numpy as np
from os import path as osp




class Solver(object):

	def __init__(self, tfrcnn, rnn, db, db_val):
		"""init

		Expects a list of 2 fully initialized tfrcnn in eval mode
		"""
		self.tfrcnn = tfrcnn
		self.rnn = rnn
		self.db = db
		self.db_val = db_val

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

	def init_weigts(self):
		pass


	def train_model(self, max_iters):
		self.data_layer = DataLayer(self.db.get_data())
		self.data_layer_val = DataLayer(self.db_val.get_data(), val=True)

		# Construct the computation graph
		self.construct_graph()

		iter = 1

		while iter < max_iters + 1:
			mb = self.data_layer.forward()
			blobs = mb['blobs']
			rois0, fc70, score0 = self.tfrcnn[0].test_image(blobs[0]['data'], blobs[0]['im_info'])
			rois1, fc71, score1 = self.tfrcnn[1].test_image(blobs[1]['data'], blobs[1]['im_info'])

			self.rnn.train_step(self.optimizer, rois0, rois1, fc70, fc71, mb['tracks'])

			#plot_correlation(mb['im_paths'], mb['blobs'][0]['im_info'], mb['blobs'][1]['im_info'], cor, r[0],r[1])

			iter += 1
			if iter % 100 == 0:
				print("Iteration: {}".format(iter))



def train_net(tfrcnn, db, db_val, max_iters=1):
	# number of inputs of rnn at the moment 300x300 correlation matrix,
	# 2x300 person scores, 300x300 IoU = 180.600
	rnn = LSTMTracker(180600, cfg.LSTM.HIDDEN_NUM, cfg.LSTM.LAYERS)
	rnn.train()
	rnn.cuda()
	solver = Solver(tfrcnn, rnn, db, db_val)
	solver.train_model(max_iters)


