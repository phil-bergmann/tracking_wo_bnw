from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import frcnn
from model.config import cfg, cfg_from_list, cfg_from_file
from frcnn_trainval import frcnn_trainval

from sacred import Experiment

from datetime import datetime
import os
import os.path as osp
import yaml

ex = Experiment()

@ex.config
def default():
	set_cfgs = None
	basenet = None
	tag =  datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	description = ""
	timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	evaluate = False

@ex.named_config
def evaluate():
	evaluate = True

# Dataset configs
@ex.named_config
def small_mot():
	imdb_name = "mot_2017_small_train"
	imdbval_name = "mot_2017_small_val"
	# around same number of epochs as voc 07 trainval (70000/5000=14)
	# 2658*14 = 37212
	# step size at max_iters/1.4
	#max_iters = 38000
	#set_cfgs = ["TRAIN.STEPSIZE", "[27000]"]
	max_iters = 110000
	set_cfgs = ["TRAIN.STEPSIZE", "[80000]"]

@ex.named_config
def mot():
	imdb_name = "mot_2017_train"
	imdbval_name = "mot_2017_small_val"
	# around same number of epochs as voc 07 trainval (70000/5000=14)
	# 5316*14 = 74424
	max_iters = 75000
	set_cfgs = ["TRAIN.STEPSIZE", "[53000]"]

@ex.named_config
def voc_basenet():
	basenet = "data/pretrained_models/voc_0712_80k-110k_converted/vgg16_faster_rcnn_iter_110000.pth"
	description = "Pretrained from VOC2007 trainval and 2012 trainval+test (80k-110k)"

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
def my_main(imdb_name, imdbval_name, max_iters, network, cfg_file, set_cfgs, weights, basenet, tag, description, evaluate, _config):

	args = {'imdb_name':imdb_name,
			'imdbval_name':imdbval_name,
			'max_iters':max_iters,
			'net':network,
			'cfg_file':None,
			'set_cfgs':None,
			'weights':weights,
			'basenet':basenet,
			'tag':tag,
			'eval':evaluate}

	print('Called with args:')
	print(args)


	# Already set everything here, so the path can be determined correctly
	if cfg_file:
		cfg_from_file(cfg_file)
	if set_cfgs:
		cfg_from_list(set_cfgs)

	# if not already present save the configuration into a file in the output folder
	outdir = osp.abspath(osp.join(cfg.ROOT_DIR, 'output', 'frcnn', cfg.EXP_DIR, imdb_name, tag))
	sacred_config = osp.join(outdir, 'sacred_config.yaml')
	if not osp.isfile(sacred_config):
		# Don't forget to make voc_basenet to None, if not resuming is not possible
		_config['basenet'] = None
		if not os.path.exists(outdir):
			os.makedirs(outdir)
		with open(sacred_config, 'w') as outfile:
			yaml.dump(_config, outfile, default_flow_style=False)

	frcnn_trainval(args)
	
