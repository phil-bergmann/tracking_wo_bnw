# setup matlpoltib to use without display
import matplotlib
matplotlib.use('Agg')

# import here all frcnn paths
import frcnn

from model.config import cfg as frcnn_cfg

import os.path as osp

from tracker.tfrcnn import TFRCNN
from tracker.mot import MOT
from tracker.lstm_tracker import LSTMTracker
from tracker.datalayer import DataLayer
from tracker.config import cfg, get_output_dir
from tracker.utils import plot_tracks_video

import torch
import numpy as np

def tracker_demo(db_demo, frcnn_weights, rnn_weights, name):

	output_dir = get_output_dir(name)
	rnn_weights = osp.join(output_dir, rnn_weights)

	db = MOT(db_demo)

	tfrcnn = TFRCNN()
	tfrcnn.create_architecture(2, tag='default',
		anchor_scales=frcnn_cfg.ANCHOR_SCALES,
		anchor_ratios=frcnn_cfg.ANCHOR_RATIOS)
	tfrcnn.eval()
	tfrcnn.cuda()
	tfrcnn.load_state_dict(torch.load(frcnn_weights))

	rnn = LSTMTracker(300*cfg.LSTM.SAMPLE_N, cfg.LSTM.HIDDEN_NUM, cfg.LSTM.LAYERS)
	rnn.eval()
	rnn.cuda()
	rnn.load_state_dict(torch.load(rnn_weights))

	data_layer = DataLayer(db.get_data())

	print("[*] Beginning evaluation...")

	for i in range(10):
		blobs = data_layer.forward()

		rois0, fc70, score0 = tfrcnn.test_image(blobs['data'][0], blobs['im_info'])
		rois1, fc71, score1 = tfrcnn.test_image(blobs['data'][1], blobs['im_info'])

		rois0.volatile = False
		fc70.volatile = False
		score0.volatile = False
		rois1.volatile = False
		fc71.volatile = False
		score1.volatile = False

		tracks = rnn.test(rois0, rois1, fc70, fc71, score0, score1, blobs)

		# filter tracks so that for every t1 != t2: t1[i] != t2[i]
		#for j in range(tracks.shape[0]-1,-1,-1):
		#	t0 = tracks[j]
		#	for k in range(j):
		#		t1 = tracks[k]
		#		if (t0[0]==t1[0]).all() or (t0[1]==t1[1]).all():
		#			tracks = np.delete(tracks, j, axis=0)
		#			break

		plot_tracks_video(blobs, tracks, osp.join(output_dir, "tracks_onebyone"+str(i)))