# setup matlpoltib to use without display
import matplotlib
matplotlib.use('Agg')

# import here all frcnn paths
import frcnn

from model.config import cfg as frcnn_cfg

from tracker.sfrcnn import FRCNN
from tracker.mot import MOT
from tracker.config import cfg, get_output_dir, get_tb_dir
from tracker.regressor import Regressor
from tracker.solver import Solver

import torch
import numpy as np
import os.path as osp



def regressor_train(name, module_name, db_train, db_val, frcnn_weights, max_epochs, seed):

	output_dir = osp.join(get_output_dir(module_name), name)
	tb_dir = osp.join(get_tb_dir(module_name), name)

	db_train = MOT(db_train)
	if db_val:
		db_val = MOT(db_val)
	frcnn = FRCNN()
	frcnn.create_architecture(2, tag='default',
		anchor_scales=frcnn_cfg.ANCHOR_SCALES,
		anchor_ratios=frcnn_cfg.ANCHOR_RATIOS)
	frcnn.eval()
	frcnn.cuda()
	frcnn.load_state_dict(torch.load(frcnn_weights))

	# build regressor and initialize with the pretrained frcnn regressor weights
	regressor = Regressor(frcnn=frcnn)
	regressor.cuda()
	regressor.train()
	regressor.load_state_dict(torch.load(frcnn_weights))

	optim_args = {	"lr": cfg.TRAIN.LEARNING_RATE,
					"weight_decay": cfg.TRAIN.WEIGHT_DECAY,
					"momentum": cfg.TRAIN.MOMENTUM }

	solver = Solver(output_dir, tb_dir, optim_args=optim_args)
	solver.train(regressor, db_train, db_val, max_epochs, 100)

#if __name__ == "__main__":
#	rnn_train()

