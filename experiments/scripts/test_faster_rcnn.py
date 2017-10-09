from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import frcnn
from model.config import cfg_from_list
from frcnn_trainval import frcnn_trainval

from sacred import Experiment

ex = Experiment()

@ex.config
def default():
	set_cfgs = None
	voc_basenet = None

# Dataset configs
@ex.named_config
def small_mot():
	imdb_name = "mot_2017_small_train"
	imdbval_name = "mot_2017_small_val"
	# around same number of epochs as voc 07 trainval (70000/5000=14)
	# 8000*14 = 112000
	max_iters = 110000
	cfg_from_list(["TRAIN.STEPSIZE", "[80000]"])

def mot():
	imdb_name = "mot_2017_train"
	imdbval_name = "mot_2017_small_val"
	# around same number of epochs as voc 07 trainval (70000/5000=14)
	# 16000*14 = 224000
	max_iters = 220000
	cfg_from_list(["TRAIN.STEPSIZE", "[150000]"])

@ex.named_config
def voc_basenet():
	voc_basenet = None

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
def my_main(imdb_name, imdbval_name, max_iters, network, cfg_file, set_cfgs, weights, voc_basenet):
	args = {'imdb_name':imdb_name,
			'imdbval_name':imdbval_name,
			'max_iters':max_iters,
			'net':network,
			'cfg_file':cfg_file,
			'set_cfgs':set_cfgs,
			'weights':weights,
			'voc_basenet':voc_basenet}
	frcnn_trainval(args)
	
