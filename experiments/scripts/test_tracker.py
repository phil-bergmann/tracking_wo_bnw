from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from sacred import Experiment
import os.path as osp
import yaml

from tracker_test import tracker_test
from tracker.config import cfg, cfg_from_list, get_output_dir

ex = Experiment()

tracker_test = ex.capture(tracker_test)

ex.add_config('output/tracker/ex-iou-n20-h500/sacred_config.yaml')

@ex.config
def default(max_iters):
	#rnn_weights = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(max_iters) + '.pth'
	rnn_weights = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(400000) + '.pth'
	db_test = 'MOT17-09-FRCNN'

@ex.automain
def my_main(CONFIG):
	# load cfg values
	cfg_from_list(CONFIG)

	tracker_test()