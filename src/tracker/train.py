
#from model.config import cfg as frcnn_cfg

from .config import cfg, get_output_dir, get_cache_dir
from .datalayer import DataLayer
from .utils import plot_correlation, bbox_overlaps, plot_tracks


import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import numpy as np
import os
from os import path as osp
import glob
import time
import pickle

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
		self.val_writer = tb.SummaryWriter(self.tb_val_dir)

		"""
		# set up multithreading
		self.tfrcnn.share_memory()
		mp.set_start_method('forkserver')
		self.con0, child0 = mp.Pipe()
		self.con1, child1 = mp.Pipe()
		self.p0 = mp.Process(target=_multithread_tfrcnn, args=(self.tfrcnn, child0))
		self.p1 = mp.Process(target=_multithread_tfrcnn, args=(self.tfrcnn, child1))
		self.p0.start()
		self.p1.start()
		"""

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
		# Construct the computation graph
		self.construct_graph()

		#_precalc_tfrcnn(self.db)
		#_precalc_tfrcnn(self.db_val)
		self.data_layer = DataLayer(_precalc_tfrcnn(self.tfrcnn, self.db))
		#self.data_layer = DataLayer(self.db.get_data())
		
		if self.db_val:
			self.data_layer_val = DataLayer(self.db_val.get_data(), val=True)

		iter = 1
		last_snapshot_iter = 0

		total_loss = 0
		cross_entropy = 0
		binary_cross_entropy = 0

		total_loss_val = 0
		cross_entropy_val = 0
		binary_cross_entropy_val = 0


		now = time.time()

		while iter < max_iters + 1:
			#print(iter)

			#blobs = self.data_layer.forward()
			blobs = self.data_layer._get_next_precalc_minibatch()['precalc']

			"""
			self.con0.send([blobs['data'][0], blobs['im_info']])
			self.con1.send([blobs['data'][1], blobs['im_info']])
			rois0, fc70, score0 = self.con0.recv()[:]
			rois1, fc71, score1 = self.con1.recv()[:]

			rois0, fc70, score0 = Variable(torch.from_numpy(rois0)).cuda(), Variable(torch.from_numpy(fc70)).cuda(), Variable(torch.from_numpy(score0)).cuda()
			rois1, fc71, score1 = Variable(torch.from_numpy(rois1)).cuda(), Variable(torch.from_numpy(fc71)).cuda(), Variable(torch.from_numpy(score1)).cuda()
			"""

			"""
			rois0, fc70, score0 = self.tfrcnn.test_image(blobs['data'][0], blobs['im_info'])
			rois1, fc71, score1 = self.tfrcnn.test_image(blobs['data'][1], blobs['im_info'])

			# score (300x2)
			rois0.volatile = False
			rois1.volatile = False
			fc70.volatile = False
			fc71.volatile = False
			score0.volatile = False
			score1.volatile = False
			"""
			
			

			
			rois0 = Variable(torch.from_numpy(blobs['rois'][0])).cuda()
			rois1 = Variable(torch.from_numpy(blobs['rois'][1])).cuda()
			fc70 = Variable(torch.from_numpy(blobs['fc7'][0])).cuda()
			fc71 = Variable(torch.from_numpy(blobs['fc7'][1])).cuda()
			score0 = Variable(torch.from_numpy(blobs['score'][0])).cuda()
			score1 = Variable(torch.from_numpy(blobs['score'][1])).cuda()
			


			losses = self.rnn.train_step(self.optimizer, rois0, rois1, fc70, fc71, score0, score1, blobs)

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

				self.writer.add_scalar("total_loss", total_loss, iter)
				self.writer.add_scalar("cross_entropy", cross_entropy, iter)
				self.writer.add_scalar("binary_cross_entropy", binary_cross_entropy, iter)

				total_loss = 0
				cross_entropy = 0
				binary_cross_entropy = 0

				if self.db_val:
					# Also calcualte validation loss
					blobs_val = self.data_layer_val.forward()

					rois0_val, fc70_val, score0_val = self.tfrcnn.test_image(blobs_val['data'][0], blobs_val['im_info'])
					rois1_val, fc71_val, score1_val = self.tfrcnn.test_image(blobs_val['data'][1], blobs_val['im_info'])

					rois0_val.volatile = False
					rois1_val.volatile = False
					fc70_val.volatile = False
					fc71_val.volatile = False
					score0_val.volatile = False
					score1_val.volatile = False

					#print("before val rnn")

					losses_val = self.rnn.get_losses(rois0_val, rois1_val, fc70_val, fc71_val, score0_val, score1_val, blobs_val)

					total_loss_val += losses_val['total_loss'].data.cpu().numpy()[0]
					cross_entropy_val += losses_val['cross_entropy'].data.cpu().numpy()[0]
					binary_cross_entropy_val += losses_val['binary_cross_entropy'].data.cpu().numpy()[0]


			if (iter % 10000 == 0) and self.db_val:
				self.val_writer.add_scalar("total_loss", total_loss_val/100, iter)
				self.val_writer.add_scalar("cross_entropy", cross_entropy_val/100, iter)
				self.val_writer.add_scalar("binary_cross_entropy", binary_cross_entropy_val/100, iter)

				total_loss_val = 0
				cross_entropy_val = 0
				binary_cross_entropy_val = 0



			if iter % cfg.TRAIN.IMAGE_ITERS == 0:
				# Print a picture of the current tracks
				tracks = self.rnn.test(rois0, rois1, fc70, fc71, score0, score1, blobs)
				im = plot_tracks(blobs, tracks)
				self.writer.add_image('train_tracks', im, iter)

				if self.db_val:
					blobs_val = self.data_layer_val.forward()

					rois0_val, fc70_val, score0_val = self.tfrcnn.test_image(blobs_val['data'][0], blobs_val['im_info'])
					rois1_val, fc71_val, score1_val = self.tfrcnn.test_image(blobs_val['data'][1], blobs_val['im_info'])

					rois0_val.volatile = False
					rois1_val.volatile = False
					fc70_val.volatile = False
					fc71_val.volatile = False
					score0_val.volatile = False
					score1_val.volatile = False

					tracks_val = self.rnn.test(rois0_val, rois1_val, fc70_val, fc71_val, score0_val, score1_val, blobs_val)
					im = plot_tracks(blobs_val, tracks_val)
					self.val_writer.add_image('val_tracks', im, iter)


			if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
				self.snapshot(iter)
				last_snapshot_iter = iter

			iter += 1

		if last_snapshot_iter != iter - 1:
			self.snapshot(iter - 1)

		self.writer.close()
		self.valwriter.close()

		"""
		self.con0.send("close")
		self.con1.send("close")
		self.p0.join()
		self.p1.join()
		"""

def _precalc_tfrcnn(model, db):
	name = db._image_set
	data = db.get_data()
	#cachefile = osp.join(get_cache_dir(), name+'.pkl')
	print("[*] Precalculating TFRCNN output")
	for i,d in enumerate(data):
		if (i+1) % 100 == 0:
			print(" - {}".format(i+1))
		blobs = DataLayer._to_blobs([d])[0]
		rois0, fc70, score0 = model.test_image(blobs['data'][0], blobs['im_info'])
		rois1, fc71, score1 = model.test_image(blobs['data'][1], blobs['im_info'])
		d['precalc'] = {}
		d['precalc']['im_info'] = blobs['im_info']
		d['precalc']['tracks'] = blobs['tracks']
		d['precalc']['im_paths'] = d['im_paths']
		d['precalc']['rois'] = []
		d['precalc']['fc7'] = []
		d['precalc']['score'] = []

		d['precalc']['rois'].append(rois0.data.cpu().numpy())
		d['precalc']['rois'].append(rois1.data.cpu().numpy())
		d['precalc']['fc7'].append(fc70.data.cpu().numpy())
		d['precalc']['fc7'].append(fc71.data.cpu().numpy())
		d['precalc']['score'].append(score0.data.cpu().numpy())
		d['precalc']['score'].append(score1.data.cpu().numpy())

	print("[*] Finished!")

	return data




def _multithread_tfrcnn(model, con):
	while True:
		message = con.recv()
		if message == "close":
			break
		else:
			rois, fc7, score = model.test_image(message[0], message[1])
			con.send([rois.data.cpu().numpy(), fc7.data.cpu().numpy(), score.data.cpu().numpy()])

