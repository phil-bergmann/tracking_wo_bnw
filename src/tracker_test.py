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
from tracker.utils import plot_sequence

import torch
import numpy as np

def tracker_test(db_test, frcnn_weights, rnn_weights, name):

	output_dir = get_output_dir(name)
	rnn_weights = osp.join(output_dir, rnn_weights)

	# at the moment expects db to contain one sequence only
	db = MOT(db_test)

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

	blobs = DataLayer._to_blobs([db.data[0]])[0]

	# for faster evaluation only recalculate second image
	rois1, fc71, score1 = tfrcnn.test_image(blobs['data'][0], blobs['im_info'])
	rois1.volatile = False
	fc71.volatile = False
	score1.volatile = False

	#all_boxes = [[[] for _ in range(num_images)]
	#	for _ in range(imdb.num_classes)]
	# results will contain 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...}
	results = []

	print("[*] Beginning evaluation...")

	for i in range(db.size-1):
		blobs = DataLayer._to_blobs([db.data[i]])[0]

		# see above
		rois0, fc70, score0 = rois1, fc71, score1
		rois1, fc71, score1 = tfrcnn.test_image(blobs['data'][1], blobs['im_info'])
		rois1.volatile = False
		fc71.volatile = False
		score1.volatile = False

		tracks = rnn.test(rois0, rois1, fc70, fc71, score0, score1, blobs)

		im_scales = blobs['im_info'][0,2]

		tracks = tracks.data.cpu().numpy() / im_scales

		# filter tracks so that for every t1 != t2: t1[i] != t2[i]
		for j in range(tracks.shape[0]-1,-1,-1):
			t0 = tracks[j]
			for k in range(j):
				t1 = tracks[k]
				if (t0[0]==t1[0]).all() or (t0[1]==t1[1]).all():
					tracks = np.delete(tracks, j, axis=0)
					break


		for t in tracks:
			# is this entry consumed?
			matched = False
			for r in results:
				if i in r.keys() and (t[0]==r[i]).all():
					# already another track took care of this?
					#if (i+1) not in r.keys():
					r[i+1] = t[1]
					# prevents that it is added to the results
					matched = True
					break
					# this track is has already entry for current i, so keep on looking
					# if there is another track that can be matched (not implemented now,
					# problems that track is added as new after that when not matched)
					#else:
					#	continue
			# ok this entry belongs to a new track
			if not matched:
				res = {i:t[0],i+1:t[1]}
				results.append(res)

		if (i+1) % 100 == 0:
			print(" - {}/{}".format(i+1, db.size))



	filtered_results = []

	# Filter out short tracks (tracks of length 2)
	length_before = len(results)
	for d in results:
		if len(d) > 2:
			filtered_results.append(d)
	length_after = len(filtered_results)
	results = filtered_results
	print("[*] Removed {} tracks of size <= 2".format(length_before-length_after))

	db._write_results_file(results, output_dir)

	plot_sequence(results, db, osp.join(output_dir, "video"))


