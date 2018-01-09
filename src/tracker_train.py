# setup matlpoltib to use without display
import matplotlib
matplotlib.use('Agg')

# import here all frcnn paths
import frcnn

from model.config import cfg as frcnn_cfg

from tracker.tfrcnn import TFRCNN
from tracker.train import Solver
from tracker.mot import MOT
from tracker.config import cfg, get_output_dir, get_tb_dir
from tracker.lstm_tracker import LSTMTracker

import torch
import numpy as np


def tracker_train(name, db_train, db_val, frcnn_weights, max_iters, seed):
	# set all seeds
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True

	db_train = MOT(db_train)
	db_val = MOT(db_val)
	#tfrcnns = [tfrcnn(),tfrcnn()]
	tfrcnn = TFRCNN()
	# Build the tfrcnn computation graphs
	# 2 classes, background and person
	tfrcnn.create_architecture(2, tag='default',
		anchor_scales=frcnn_cfg.ANCHOR_SCALES,
		anchor_ratios=frcnn_cfg.ANCHOR_RATIOS)
	tfrcnn.eval()
	tfrcnn.cuda()
	tfrcnn.load_state_dict(torch.load(frcnn_weights))

	output_dir = get_output_dir(name)
	tb_dir = get_tb_dir(name)

	# number of inputs of rnn at the moment 300x300 correlation matrix,
	# 2x300 person scores, 300x300 IoU = 180.600
	# NEW: input now downsampled to 300xN, so with N=20 now 6000
	rnn = LSTMTracker(300*cfg.LSTM.SAMPLE_N, cfg.LSTM.HIDDEN_NUM, cfg.LSTM.LAYERS)
	rnn.train()
	rnn.cuda()
	solver = Solver(tfrcnn, rnn, db_train, db_val, output_dir, tb_dir)
	solver.train_model(max_iters)

#if __name__ == "__main__":
#	rnn_train()

