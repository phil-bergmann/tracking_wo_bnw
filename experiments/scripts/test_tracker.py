import _init_paths

import os.path as osp
import os
import numpy as np
import yaml
import time

from sacred import Experiment
import torch
from torch.utils.data import DataLoader

from model.config import cfg as frcnn_cfg

from tracker.config import get_output_dir, get_tb_dir
from tracker.sfrcnn import FRCNN
from tracker.lstm_regressor import LSTM_Regressor
#from tracker.appearance_lstm import Appearance_LSTM
from tracker.mot_sequence import MOT_Sequence
from tracker.tracker import Tracker
from tracker.utils import plot_sequence

test = ["MOT17-01", "MOT17-03", "MOT17-06", "MOT17-07", "MOT17-08", "MOT17-12", "MOT17-14"]
train = ["MOT17-13", "MOT17-11", "MOT17-10", "MOT17-09", "MOT17-05", "MOT17-04", "MOT17-02", ]
sequences = ["MOT17-09"]
sequences = train

ex = Experiment()

ex.add_config('experiments/cfgs/tracker.yaml')
ex.add_config('output/tracker/lstm_regressor/reg-6dead-srf15/sacred_config.yaml')

LSTM_Regressor = ex.capture(LSTM_Regressor, prefix='lstm_regressor.lstm_regressor')
#Appearance_LSTM = ex.capture(Appearance_LSTM, prefix='lstm_regressor.appearance_lstm')
Tracker = ex.capture(Tracker, prefix='tracker.tracker')

@ex.automain
def my_main(lstm_regressor, tracker, _config):
	print(_config)

	lstm_regressor_dir = osp.join(get_output_dir(lstm_regressor['module_name']), lstm_regressor['name'])

	# save sacred config to experiment
	output_dir = osp.join(get_output_dir(tracker['module_name']), tracker['name'])
	sacred_config = osp.join(output_dir, 'sacred_config.yaml')
	
	if not osp.exists(output_dir):
		os.makedirs(output_dir)
	with open(sacred_config, 'w') as outfile:
		yaml.dump(_config, outfile, default_flow_style=False)

	##########################
	# Initialize the modules #
	##########################
	print("[*] Building FRCNN")

	frcnn = FRCNN()
	frcnn.create_architecture(2, tag='default',
		anchor_scales=frcnn_cfg.ANCHOR_SCALES,
		anchor_ratios=frcnn_cfg.ANCHOR_RATIOS)
	frcnn.eval()
	frcnn.cuda()
	frcnn.load_state_dict(torch.load(lstm_regressor['frcnn_weights']))

	
	print("[*] Building Regressor")
	regressor = LSTM_Regressor()
	regressor.cuda()
	regressor.eval()
	regressor.load_state_dict(torch.load(tracker['regressor_weights']))

	print("[*] Building Tracker")
	tracker = Tracker(frcnn, regressor)
	
	####################
	# Begin evaluation #
	####################
	print("[*] Begin Evaluation")

	for s in sequences:
		now = time.time()

		print("[*] Evaluating: {}".format(s))
		tracker.reset()
		db = MOT_Sequence(s)
		dl = DataLoader(db, batch_size=1, shuffle=False)

		for sample in dl:
			tracker.step(sample)

		results = tracker.get_results()
		print("Tracks found: {}".format(len(results)))
		print("[*] Time needed for {} evaluation: {:.3f} s".format(s, time.time() - now))

		db.write_results(results, output_dir)

		plot_sequence(results, db, osp.join(output_dir, s))

