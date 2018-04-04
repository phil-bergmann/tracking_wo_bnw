import _init_paths

from sacred import Experiment
import os.path as osp
import os
import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from model.config import cfg as frcnn_cfg

from tracker.config import get_output_dir, get_tb_dir
from tracker.solver import Solver
from tracker.sfrcnn import FRCNN
from tracker.mot_wrapper import MOT_Wrapper
from tracker.simple_regressor import Simple_Regressor


ex = Experiment()

ex.add_config('experiments/cfgs/simple_regressor.yaml')
MOT_Wrapper = ex.capture(MOT_Wrapper, prefix='simple_regressor')
Simple_Regressor = ex.capture(Simple_Regressor, prefix='simple_regressor.simple_regressor')
Solver = ex.capture(Solver, prefix='simple_regressor.solver')

@ex.automain
def my_main(simple_regressor, _config):
	# set all seeds
	torch.manual_seed(simple_regressor['seed'])
	torch.cuda.manual_seed(simple_regressor['seed'])
	np.random.seed(simple_regressor['seed'])
	torch.backends.cudnn.deterministic = True

	print(_config)

	output_dir = osp.join(get_output_dir(simple_regressor['module_name']), simple_regressor['name'])
	tb_dir = osp.join(get_tb_dir(simple_regressor['module_name']), simple_regressor['name'])

	# save sacred config to experiment
	sacred_config = osp.join(output_dir, 'sacred_config.yaml')

	
	if not osp.exists(output_dir):
		os.makedirs(output_dir)
	with open(sacred_config, 'w') as outfile:
		yaml.dump(_config, outfile, default_flow_style=False)

	#########################
	# Initialize dataloader #
	#########################
	print("[*] Initializing Dataloader")

	db_train = MOT_Wrapper(simple_regressor['db_train'])
	#db_train = DataLoader(db_train, batch_size=1, shuffle=True)

	if simple_regressor['db_val']:
		db_val = MOT_Wrapper(simple_regressor['db_val'])
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
	frcnn.load_state_dict(torch.load(simple_regressor['frcnn_weights']))
	
	# build lstm regressor
	print("[*] Building Regressor")
	regressor = Simple_Regressor()
	regressor.cuda()
	regressor.train()

	# precalcuate conv features
	if simple_regressor['precalculate_features']:
		db_train.precalculate_conv(frcnn)
	sampler = WeightedRandomSampler(db_train.weights, 30000, True)
	db_train = DataLoader(db_train, batch_size=1, shuffle=False, sampler=sampler)

	##################
	# Begin training #
	##################
	print("[*] Solving ...")

	model_args = {'frcnn':frcnn}

	solver = Solver(output_dir, tb_dir)
	solver.train(regressor, db_train, db_val, simple_regressor['max_epochs'], 100, model_args=model_args)
	
	
	