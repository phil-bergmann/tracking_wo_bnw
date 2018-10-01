from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from model.config import cfg, cfg_from_list, cfg_from_file
from frcnn_demo import frcnn_demo

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
	score_thresh = 0.05
	clip_bbox = False

# Image files to take
@ex.named_config
def demo():
	im_names = ['004545.jpg', 'tr-02-000260.jpg', 'tr-04-000424.jpg', 'tr-05-000703.jpg',
				'tr-09-000262.jpg', 'tr-10-000058.jpg', 'tr-11-000686.jpg', 'tr-13-000510.jpg',
				'te-01-000136.jpg', 'te-03-000886.jpg', 'te-06-000897.jpg', 'te-07-000333.jpg',
				'te-08-000496.jpg', 'te-12-000504.jpg', 'te-14-000509.jpg', '000153.png',
				'000154.png', '000156.png', '000161.png']
	
@ex.automain
def my_main(imdb_name, network, cfg_file, set_cfgs, tag, max_iters, im_names, score_thresh, clip_bbox):

	# Clip bboxes after bbox reg to image boundary
	cfg_from_list(['TEST.BBOX_CLIP', str(clip_bbox)])

	# Already set everything here, so the path can be determined correctly
	if cfg_file:
		cfg_from_file(cfg_file)
	if set_cfgs:
		cfg_from_list(set_cfgs)

	model_dir = osp.abspath(osp.join(cfg.ROOT_DIR, 'output', 'frcnn', cfg.EXP_DIR,
        imdb_name, tag))
	model = osp.join(model_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(max_iters) + '.pth')
	output_dir = osp.join(model_dir, 'demo')
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	args = {'imdb_name':imdb_name,
			'net':network,
			'cfg_file':None,
			'set_cfgs':None,
			'tag':tag,
			'output_dir':output_dir,
			'model':model,
			'im_names':im_names,
			'score_thresh':score_thresh}

	print('Called with args:')
	print(args)

	frcnn_demo(args)