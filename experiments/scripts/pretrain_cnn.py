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
from tracker.alex import alex

ex = Experiment()
ex.add_config('experiments/cfgs/pretrain_cnn.yaml')

MOT_Siamese_Wrapper = ex.capture(MOT_Siamese_Wrapper, prefix='cnn')
Solver = ex.capture(Solver, prefix='cnn.solver')
#alex = ex.capture(alex, prefix='cnn.cnn')

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

	db_train = MOT_Siamese_Wrapper(cnn['db_train'])
	db_train = DataLoader(db_train, batch_size=1, shuffle=True)

	#if cnn['db_val']:
	#	db_val = MOT_Wrapper(cnn['db_val'], MOT_Tracks)
	#	db_val = DataLoader(db_val, batch_size=1, shuffle=True)
	#else:
	#	db_val = None
	db_val = None

	#for i,d in enumerate(db_train,1):
	#	if i == 1:
	#		print(d[1][0])
	#	for j,im in enumerate(d[0][0],1):
	#		cv2.imwrite(osp.join(output_dir, str(j)+'.png'),im.numpy())

	
	##########################
	# Initialize the modules #
	##########################
	print("[*] Building CNN")
	network = alex(pretrained=True, **cnn['cnn'])
	print(network.output_dim)
	network.train()
	network.cuda()

	##################
	# Begin training #
	##################
	print("[*] Solving ...")

	solver = Solver(output_dir, tb_dir)
	solver.train(network, db_train, db_val, cnn['max_epochs'], 100, model_args=cnn['model_args'])
	
	