import _init_paths

from sacred import Experiment
import os.path as osp
import os
import numpy as np
import yaml
import cv2

import torch
from torch.utils.data import DataLoader

from tracker.config import get_output_dir, get_tb_dir
from tracker.solver import Solver
from tracker.mot_siamese_wrapper import MOT_Siamese_Wrapper
from tracker.resnet import resnet50

ex = Experiment()
ex.add_config('experiments/cfgs/pretrain_cnn.yaml')

Solver = ex.capture(Solver, prefix='cnn.solver')

@ex.automain
def my_main(_config, cnn):
	# set all seeds
	torch.manual_seed(cnn['seed'])
	torch.cuda.manual_seed(cnn['seed'])
	np.random.seed(cnn['seed'])
	torch.backends.cudnn.deterministic = True

	print(_config)

	output_dir = osp.join(get_output_dir(cnn['module_name']), cnn['name'])
	tb_dir = osp.join(get_tb_dir(cnn['module_name']), cnn['name'])

	sacred_config = osp.join(output_dir, 'sacred_config.yaml')

	if not osp.exists(output_dir):
		os.makedirs(output_dir)
	with open(sacred_config, 'w') as outfile:
		yaml.dump(_config, outfile, default_flow_style=False)

	#########################
	# Initialize dataloader #
	#########################
	print("[*] Initializing Dataloader")

	dataloader = cnn['dataloader']
	db_train = MOT_Siamese_Wrapper(cnn['db_train'], dataloader)
	db_train = DataLoader(db_train, batch_size=1, shuffle=True)

	if cnn['db_val']:
		dataloader['split'] = "small_val"
		db_val = MOT_Siamese_Wrapper(cnn['db_val'], dataloader)
		db_val = DataLoader(db_val, batch_size=1, shuffle=True)
	else:
		db_val = None
	
	##########################
	# Initialize the modules #
	##########################
	print("[*] Building CNN")
	network = resnet50(pretrained=True, **cnn['cnn'])
	network.train()
	network.cuda()

	##################
	# Begin training #
	##################
	print("[*] Solving ...")
	
	if cnn['lr_scheduler']:
		# build scheduling like in "In Defense of the Triplet Loss for Person Re-Identification"
		# from Hermans et al.
		lr = cnn['solver']['optim_args']['lr']
		iters_per_epoch = len(db_train)
		# we want to keep lr until iter 15000 and from there to iter 25000 a exponential decay
		l = eval("lambda epoch: 1 if epoch*{} < 15000 else 0.001**((epoch*{} - 15000)/(25000-15000))".format(
																iters_per_epoch,  iters_per_epoch))
	else:
		l = None
	solver = Solver(output_dir, tb_dir, lr_scheduler_lambda=l)
	solver.train(network, db_train, db_val, cnn['max_epochs'], 100, model_args=cnn['model_args'])
	
	
	
	