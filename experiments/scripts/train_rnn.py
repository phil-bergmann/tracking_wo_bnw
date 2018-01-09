from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from sacred import Experiment
import os.path as osp
import yaml

from tracker_train import tracker_train
from tracker.config import cfg, cfg_from_list, get_output_dir

ex = Experiment()



#Solver.train = ex.capture(Solver.train, prefix='nn_train.solver')
tracker_train = ex.capture(tracker_train)

ex.add_config('experiments/cfgs/tracker.yaml')

@ex.automain
def my_main(CONFIG, name, _config):

	# load cfg values
	cfg_from_list(CONFIG)
	print(_config)

	# save sacred config to experiment
	# if not already present save the configuration into a file in the output folder
	outdir = get_output_dir(name)
	sacred_config = osp.join(outdir, 'sacred_config.yaml')
	if not osp.isfile(sacred_config):
		with open(sacred_config, 'w') as outfile:
			yaml.dump(_config, outfile, default_flow_style=False)

	tracker_train()
