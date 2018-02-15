import _init_paths

from sacred import Experiment
import os.path as osp
import os
import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader

from model.config import cfg as frcnn_cfg

from tracker.config import get_output_dir, get_tb_dir
from tracker.solver import Solver
from tracker.sfrcnn import FRCNN
from tracker.lstm_regressor import LSTM_Regressor
from tracker.mot_wrapper import MOT_Wrapper
from tracker.mot_tracks import MOT_Tracks


ex = Experiment()

ex.add_config('experiments/cfgs/lstm_regressor.yaml')

LSTM_Regressor = ex.capture(LSTM_Regressor, prefix='lstm_regressor.lstm_regressor')
Solver = ex.capture(Solver, prefix='lstm_regressor.solver')

@ex.automain
def my_main(lstm_regressor, _config):
	# set all seeds
	torch.manual_seed(lstm_regressor['seed'])
	torch.cuda.manual_seed(lstm_regressor['seed'])
	np.random.seed(lstm_regressor['seed'])
	torch.backends.cudnn.deterministic = True

	print(_config)

	output_dir = osp.join(get_output_dir(lstm_regressor['module_name']), lstm_regressor['name'])
	tb_dir = osp.join(get_tb_dir(lstm_regressor['module_name']), lstm_regressor['name'])

	# save sacred config to experiment
	# if already present throw an error
	sacred_config = osp.join(output_dir, 'sacred_config.yaml')

	#assert not osp.isfile(sacred_config), "[!] Config already present: {}".format(sacred_config)
	
	if not osp.exists(output_dir):
		os.makedirs(output_dir)
	with open(sacred_config, 'w') as outfile:
		yaml.dump(_config, outfile, default_flow_style=False)

	#########################
	# Initialize dataloader #
	#########################
	print("[*] Initializing Dataloader")

	db_train = MOT_Wrapper(lstm_regressor['db_train'], MOT_Tracks)
	db_train = DataLoader(db_train, batch_size=1, shuffle=True)

	if lstm_regressor['db_val']:
		db_val = MOT_Wrapper(lstm_regressor['db_val'], MOT_Tracks)
		db_val = DataLoader(db_val, batch_size=1, shuffle=True)
	else:
		db_val = None

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

	
	# build lstm regressor
	print("[*] Building Regressor")
	regressor = LSTM_Regressor()
	regressor.cuda()
	regressor.train()

	
	##################
	# Begin training #
	##################
	print("[*] Solving ...")

	solver = Solver(output_dir, tb_dir)
	solver.train(regressor, frcnn, db_train, db_val, lstm_regressor['max_epochs'], 100)
	