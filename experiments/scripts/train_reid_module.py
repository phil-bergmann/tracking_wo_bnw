import _init_paths

from sacred import Experiment
import os.path as osp
import os
import numpy as np
import yaml

import torch

from tracker.config import get_output_dir, get_tb_dir
from tracker.reid_module import ReID_Module
from tracker.solver import Solver
from tracker.mot_siamese_wrapper import MOT_Siamese_Wrapper
from torch.utils.data import DataLoader
from tracker.alex import Alex

ex = Experiment()

ex.add_config('experiments/cfgs/reid_module.yaml')
ex.add_config('output/tracker/pretrain_cnn/alex15/sacred_config.yaml')


ReID_Module = ex.capture(ReID_Module, prefix='reid_module.reid_module')
MOT_Siamese_Wrapper = ex.capture(MOT_Siamese_Wrapper, prefix='reid_module')
Solver = ex.capture(Solver, prefix='reid_module.solver')
Alex = ex.capture(Alex, prefix='cnn.cnn')

@ex.automain
def my_main(reid_module, _config):
	# set all seeds
	torch.manual_seed(reid_module['seed'])
	torch.cuda.manual_seed(reid_module['seed'])
	np.random.seed(reid_module['seed'])
	torch.backends.cudnn.deterministic = True

	print(_config)

	output_dir = osp.join(get_output_dir(reid_module['module_name']), reid_module['name'])
	tb_dir = osp.join(get_tb_dir(reid_module['module_name']), reid_module['name'])

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

	db_train = MOT_Siamese_Wrapper(reid_module['db_train'])
	db_train = DataLoader(db_train, batch_size=1, shuffle=True)

	db_val = None

	##########################
	# Initialize the modules #
	##########################
	print("[*] Building CNN")
	
	network = Alex()
	network.load_state_dict(torch.load(reid_module['cnn_weights']))
	network.train()
	network.cuda()
	
	print("[*] Building ReID Module")

	module = ReID_Module(cnn=network, mode="TRAIN")
	module.train()
	module.cuda()

	solver = Solver(output_dir, tb_dir)
	solver.train(module, db_train, db_val, reid_module['max_epochs'], 100, model_args=reid_module['model_args'])