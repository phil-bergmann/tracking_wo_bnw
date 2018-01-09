from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from sacred import Experiment
import os.path as osp
import yaml

from tracker_demo import tracker_demo
from tracker.config import cfg, cfg_from_list, get_output_dir

ex = Experiment()

tracker_demo = ex.capture(tracker_demo)

ex.add_config('output/tracker/test5-pos/sacred_config.yaml')

@ex.config
def default(max_iters):
	rnn_weights = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(max_iters) + '.pth'
	db_demo = 'small_train'

@ex.automain
def my_main(CONFIG):
	# load cfg values
	cfg_from_list(CONFIG)

	tracker_demo()