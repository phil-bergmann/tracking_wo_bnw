from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import frcnn
from model.config import cfg, cfg_from_list, cfg_from_file
from frcnn_test import frcnn_test

from sacred import Experiment

import os
import os.path as osp

ex = Experiment()

@ex.config
def default():
	cfg_file = None
	set_cfgs = None
	comp = False
	max_per_image = 100
	basenet = None
	tag =  ""
	description = ""
	timestamp = ""
	imdbval_name = ""
	imdb_name = ""
	max_iters = 0
	description = ""
	weights = ""
	network = ""
	evaluate = False

# Dataset configs
@ex.named_config
def small_mot():
	imdbtest_name = "mot_2017_small_val"

@ex.named_config
def mot():
	imdbtest_name = "mot_2017_all"

@ex.named_config
def mot_test():
	imdbtest_name = "mot_2017_test"

#@ex.named_config
#def res101():
#	network = "res101"
#	weights = "data/imagenet_weights/{}.pth".format(network)
#	cfg_file = "experiments/cfgs/{}.yml".format(network)

#@ex.named_config
#def vgg16():
#	network = "vgg16"
#	weights = "data/imagenet_weights/{}.pth".format(network)
#	cfg_file = "experiments/cfgs/{}.yml".format(network)
	

@ex.automain
def my_main(imdb_name, imdbtest_name, network, cfg_file, set_cfgs, tag, comp, max_iters, max_per_image):

	# Already set everything here, so the path can be determined correctly
	if cfg_file:
		cfg_from_file(cfg_file)
	if set_cfgs:
		cfg_from_list(set_cfgs)

	model_dir = osp.abspath(osp.join(cfg.ROOT_DIR, 'output', 'frcnn', cfg.EXP_DIR,
        imdb_name, tag))
	model = osp.join(model_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(max_iters) + '.pth')
	output_dir = osp.join(model_dir, imdbtest_name)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	args = {#'imdb_name':imdb_name,
			'imdbtest_name':imdbtest_name,
			'net':network,
			'cfg_file':None,
			'set_cfgs':None,
			'tag':tag,
			'comp_mode':comp,
			'max_per_image':max_per_image,
			'output_dir':output_dir,
			'model':model}

	print('Called with args:')
	print(args)

	frcnn_test(args)