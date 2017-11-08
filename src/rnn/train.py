
#from model.config import cfg as frcnn_cfg

from .config import cfg
from .datalayer import DataLayer

import torch



class Solver(object):

	def __init__(self, tfrcnn, rnn, db, db_val):
		"""init

		Expects a list of 3 fully initialized tfrcnn in eval mode
		"""
		self.tfrcnn = tfrcnn
		self.rnn = rnn
		self.db = db
		self.db_val = db_val

	def construct_graph(self):
		# Set the random seed
		torch.manual_seed(cfg.RNG_SEED)
		

		lr = cfg.TRAIN.LEARNING_RATE

		#self.optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)


	def train_model(self, max_iters):
		self.data_layer = DataLayer(self.db.get_data())
		self.data_layer_val = DataLayer(self.db_val.get_data(), val=True)

		# Construct the computation graph
		self.construct_graph()

		iter = 1

		while iter < max_iters + 1:
			mb = self.data_layer.forward()
			for i in range(3):
				blob = mb['blobs'][i]
				rois, fc7 = self.tfrcnn[i].test_image(blob['data'], blob['im_info'])
				#print(rois.size())
				#print(fc7.size())
			iter += 1
			if iter % 100 == 0:
				print("Iteration: {}".format(iter))


def train_net(tfrcnn, rnn, db, db_val, max_iters=10000):
	solver = Solver(tfrcnn, rnn, db, db_val)
	solver.train_model(max_iters)


