import frcnn

from model.test import _get_blobs
from model.config import cfg as frcnn_cfg

from rnn.tfrcnn import tfrcnn
from rnn.train import train_net
from rnn.mot import MOT

import torch


FRCNN_WEIGHTS = "/usr/stud/bergmanp/sequential_tracking/output/frcnn/vgg16/mot_2017_train/stop_180k_allBB/vgg16_faster_rcnn_iter_180000.pth"

def rnn_train():
	db = MOT("small_train")
	db_val = MOT("small_val")
	tfrcnns = [tfrcnn(),tfrcnn(),tfrcnn()]
	# Build the tfrcnn computation graphs
	for x in tfrcnns:
		x.create_architecture(1, tag='default',
			anchor_scales=frcnn_cfg.ANCHOR_SCALES,
			anchor_ratios=frcnn_cfg.ANCHOR_RATIOS)
		x.eval()
		x.cuda()
		x.load_state_dict(torch.load(FRCNN_WEIGHTS))

	train_net(tfrcnns,None,db,db_val)

if __name__ == "__main__":
	rnn_train()

