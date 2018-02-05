from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from sacred import Experiment
import os.path as osp
import yaml

from regressor_train import regressor_train
from tracker.config import cfg, cfg_from_list, get_output_dir

ex = Experiment()

regressor_train = ex.capture(regressor_train)

#ex.add_config('experiments/cfgs/tracker.yaml')
@ex.config
def default():
	frcnn_weights= '/usr/stud/bergmanp/sequential_tracking/output/frcnn/vgg16/mot_2017_train/stop_180k_allBB/vgg16_faster_rcnn_iter_180000.pth'
	db_train = 'small_train'
	db_val = 'small_val'
	max_epochs = 7
	seed = 12345
	name = 'small_7_onlyBoxLoss'
	module_name = 'regressor'

@ex.automain
def my_main(name, _config):

	# load cfg values
	#cfg_from_list(CONFIG)
	print(_config)
	# save sacred config to experiment
	# if not already present save the configuration into a file in the output folder
	#outdir = get_output_dir(name)
	#sacred_config = osp.join(outdir, 'sacred_config.yaml')
	#if not osp.isfile(sacred_config):
	#	with open(sacred_config, 'w') as outfile:
	#		yaml.dump(_config, outfile, default_flow_style=False)

	regressor_train()
