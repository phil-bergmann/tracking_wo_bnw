# Script that analyses the visibiities of persons in the dataset

import _init_paths

import os.path as osp
import os
import numpy as np
from scipy import ndimage
import yaml
import time
import csv
import matplotlib.pyplot as plt

from sacred import Experiment

from tracker.config import get_output_dir
from tracker.mot_sequence import MOT_Sequence

ex = Experiment()

test = ["MOT17-01", "MOT17-03", "MOT17-06", "MOT17-07", "MOT17-08", "MOT17-12", "MOT17-14"]
train = ["MOT17-13", "MOT17-11", "MOT17-10", "MOT17-09", "MOT17-05", "MOT17-04", "MOT17-02", ]
sequences = train


@ex.automain
def my_main(_config):
	print(_config)
	
	##################
	# Begin analysis #
	##################
	print("[*] Beginning mean/variance analysis ..")
	print("[!] Warning: All in BGR order!")


	mean = []
	mean_sqr = []
	var = []

	for s in sequences:

		print("[*] Analysing: {}".format(s))
		db = MOT_Sequence(s)
		for batch in db:
			image = batch['or_data'][0] / 255
			image_sqr = image**2
			mean.append(np.mean(image, axis=(0,1)))
			mean_sqr.append(np.mean(image_sqr, axis=(0,1)))
		print("Mean: {}".format(np.stack(mean,0).mean(0)))
		print("Std: {}".format(np.sqrt(np.stack(mean_sqr,0).mean(0) - np.stack(mean,0).mean(0)**2)))
	print("[*] Whole Dataset")
	print("Mean: {}".format(np.stack(mean,0).mean(0)))
	print("Std: {}".format(np.sqrt(np.stack(mean_sqr,0).mean(0) - np.stack(mean,0).mean(0)**2)))



