from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from sacred import Experiment
import os.path as osp
import yaml

from simple_tracker_demo import simple_tracker_demo
from tracker.config import cfg, cfg_from_list, get_output_dir

ex = Experiment()

simple_tracker_demo = ex.capture(simple_tracker_demo)

#ex.add_config('output/tracker/test5-pos/sacred_config.yaml')

@ex.config
def default():
	db_demo = 'MOT17-04-FRCNN'
	frcnn_weights= '/usr/stud/bergmanp/sequential_tracking/output/frcnn/vgg16/mot_2017_train/stop_180k_allBB/vgg16_faster_rcnn_iter_180000.pth'
	regressor_weights = '/usr/stud/bergmanp/sequential_tracking/output/tracker/regressor/small_7_onlyBoxLoss/Regressor_v0.1_iter_12558.pth'

@ex.automain
def my_main():
	# load cfg values
	#cfg_from_list(CONFIG)

	simple_tracker_demo()