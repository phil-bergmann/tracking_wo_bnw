from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from model.config import cfg, cfg_from_list, cfg_from_file
from frcnn_trainval import frcnn_trainval

from sacred import Experiment

from datetime import datetime
import os
import os.path as osp
import yaml

ex = Experiment()

frcnn_trainval = ex.capture(frcnn_trainval)

@ex.config
def default():
	set_cfgs = None
	tag =  datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	description = ""
	timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Dataset configs
@ex.named_config
def small_mot():
	imdb_name = "mot_2017_small_train"
	imdbval_name = "mot_2017_small_val"
	max_iters = 180000
	set_cfgs = ["TRAIN.STEPSIZE", "[125000]"]

@ex.named_config
def mot():
	imdb_name = "mot_2017_train"
	imdbval_name = "mot_2017_small_val"
	max_iters = 180000
	set_cfgs = ["TRAIN.STEPSIZE", "[125000]"]

@ex.named_config
def res101():
	network = "res101"
	weights = "data/imagenet_weights/{}.pth".format(network)
	cfg_file = "experiments/cfgs/{}.yml".format(network)

@ex.named_config
def vgg16():
	network = "vgg16"
	weights = "data/imagenet_weights/{}.pth".format(network)
	cfg_file = "experiments/cfgs/{}.yml".format(network)
	

@ex.automain
def my_main(tag, cfg_file, set_cfgs, imdb_name, _config):

	# Already set everything here, so the path can be determined correctly
	if cfg_file:
		cfg_from_file(cfg_file)
	if set_cfgs:
		cfg_from_list(set_cfgs)

	print('Called with args:')
	print(_config)

	# if not already present save the configuration into a file in the output folder
	outdir = osp.abspath(osp.join(cfg.ROOT_DIR, 'output', 'frcnn', cfg.EXP_DIR, imdb_name, tag))
	sacred_config = osp.join(outdir, 'sacred_config.yaml')
	if not osp.isfile(sacred_config):
		if not os.path.exists(outdir):
			os.makedirs(outdir)
		with open(sacred_config, 'w') as outfile:
			yaml.dump(_config, outfile, default_flow_style=False)

	frcnn_trainval()