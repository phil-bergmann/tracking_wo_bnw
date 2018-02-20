# Script that analyses the displacements of persons from one frame to the other in multiples of width/height of BB

import _init_paths

import os.path as osp
import os
import numpy as np
import yaml
import time
import csv
import matplotlib.pyplot as plt
from numpy.linalg import norm

from sacred import Experiment
from torch.utils.data import DataLoader

from tracker.config import get_output_dir
from tracker.mot_sequence import MOT_Sequence

test = ["MOT17-01", "MOT17-03", "MOT17-06", "MOT17-07", "MOT17-08", "MOT17-12", "MOT17-14"]
train = ["MOT17-13", "MOT17-11", "MOT17-10", "MOT17-09", "MOT17-05", "MOT17-04", "MOT17-02", ]
sequences = train

ex = Experiment()

@ex.automain
def my_main(_config):
	print(_config)

	# save sacred config to experiment
	output_dir = osp.join(get_output_dir('dataset'), 'displacement')
	
	if not osp.exists(output_dir):
		os.makedirs(output_dir)
	
	##################
	# Begin analysis #
	##################
	print("[*] Beginning visibility analysis ..")

	num_range = np.arange(0.0,1.0,0.1)

	res = {}
	res['ges'] = {'x/w':[], 'y/w':[], 'd/w':[], 'x/h':[], 'y/h':[], 'd/h':[]}

	for s in sequences:
		res[s] = {'x/w':[], 'y/w':[], 'd/w':[], 'x/h':[], 'y/h':[], 'd/h':[]}

		print("[*] Analysing: {}".format(s))
		db = MOT_Sequence(s, vis_threshold=0.0)

		for i in range(len(db)-1):
			gt0 = db[i]['gt']
			gt1 = db[i+1]['gt']
			for k,v in gt0.items():
				if k in gt1.keys():
					x10 = v[0]
					y10 = v[1]
					x20 = v[2]
					y20 = v[3]
					cx0 = (x10+x20)/2
					cy0 = (y10+y20)/2
					w = x20-x10+1
					h = y20-y10+1

					x11 = gt1[k][0]
					y11 = gt1[k][1]
					x21 = gt1[k][2]
					y21 = gt1[k][3]
					cx1 = (x11+x21)/2
					cy1 = (y11+y21)/2

					dx = abs(cx1-cx0)
					dy = abs(cy1-cy0)
					dd = norm([dx,dy])

					res[s]['x/w'].append(dx/w)
					res[s]['y/w'].append(dy/w)
					res[s]['d/w'].append(dd/w)
					res[s]['x/h'].append(dx/h)
					res[s]['y/h'].append(dy/h)
					res[s]['d/h'].append(dd/h)

					res['ges']['x/w'].append(dx/w)
					res['ges']['y/w'].append(dy/w)
					res['ges']['d/w'].append(dd/w)
					res['ges']['x/h'].append(dx/h)
					res['ges']['y/h'].append(dy/h)
					res['ges']['d/h'].append(dd/h)


	"""with open(osp.join(output_dir, 'results.csv'), 'w', newline='') as csvfile:
		fieldnames = ['seq_name', '0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for k,v in res.items():
			line = {fieldnames[0]: k}
			for i in range(1,len(fieldnames)):
				line[fieldnames[i]] = 0
			for i in v:
				ind = np.where(i >= num_range)[0][-1]
				line[fieldnames[ind+1]] += 1
			writer.writerow(line)"""

	for k,v in res.items():
		for kk,vv in v.items():
			plt.hist(vv,bins=20, rwidth=0.95)
			mean = np.mean(vv)
			plt.title("Displacement Histogram {} {} (mean: {:.3f})".format(k,kk,mean))
			plt.xlabel("Value")
			plt.ylabel("Occurence")
			plt.draw()
			plt.savefig(osp.join(output_dir, k+'-'+kk[0]+kk[2]+'.jpg'), dpi=300)
			plt.close()


