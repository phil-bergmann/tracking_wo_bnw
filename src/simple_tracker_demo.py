# setup matlpoltib to use without display
import matplotlib
matplotlib.use('Agg')

# import here all frcnn paths
import frcnn

from model.config import cfg as frcnn_cfg


import os.path as osp

from tracker.sfrcnn import FRCNN
from tracker.mot import MOT
from tracker.lstm_tracker import LSTMTracker
from tracker.datalayer import DataLayer
from tracker.config import cfg, get_output_dir
from tracker.utils import plot_simple
from tracker.utils import plot_sequence
from tracker.frcnn_tracker import FRCNN_TRACKER
from tracker.regressor import Regressor

import torch
from torch.autograd import Variable
import numpy as np

out = "regressor0.1_small7_onlyBoxLoss"

def simple_tracker_demo(db_demo, frcnn_weights):

	output_dir = get_output_dir('simple_tracker')

	db = MOT(db_demo)

	frcnn = FRCNN()
	frcnn.create_architecture(2, tag='default',
		anchor_scales=frcnn_cfg.ANCHOR_SCALES,
		anchor_ratios=frcnn_cfg.ANCHOR_RATIOS)
	frcnn.eval()
	frcnn.cuda()
	frcnn.load_state_dict(torch.load(frcnn_weights))

	#regressor = Regressor(frcnn=frcnn)
	#regressor.cuda()
	#regressor.eval()
	#regressor.load_state_dict(torch.load(regressor_weights))

	print("[*] Beginning evaluation...")

	"""
	#data_layer = DataLayer(db.get_data())
	blobs = DataLayer._to_blobs([db.data[0]])[0]

	"""
	#test = ["MOT17-01", "MOT17-03", "MOT17-06", "MOT17-07", "MOT17-08", "MOT17-12", "MOT17-14"]
	test = ["MOT17-09"]

	for t in test:
		tracker = FRCNN_TRACKER(frcnn)

		print("[*] Evaluating: {}".format(t))
		db = MOT(t+"-FRCNN")
		for i in range(db.size-1):
			blobs = DataLayer._to_blobs([db.data[i]])[0]
			tracker.step(blobs)
		blobs = DataLayer._to_blobs([db.data[i]])[0]
		tracker.step(blobs)
		results = tracker.get_results()

		results = list(results.values())

		print("Tracks found: {}".format(len(results)))

		db._write_results_file(results, osp.join(output_dir, out))
		
		plot_sequence(results, db, osp.join(output_dir, out, t))
