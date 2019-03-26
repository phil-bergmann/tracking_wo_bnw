import numpy as np
import cv2
import os
import os.path as osp
import configparser
import csv
import re
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Compose, ToTensor

from frcnn.model import test

from ..config import cfg


class MOT15_Sequence(Dataset):
	"""Loads a sequence from the 2DMOT15 dataset.

	This dataloader is designed so that it can handle only one sequence, if more have to be handled
	at once one should use a wrapper class.
	"""

	def __init__(self, seq_name=None, vis_threshold=0.0, normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225]):
		"""
		Args:
			seq_name (string): Sequence to take
			vis_threshold (float): Threshold of visibility of persons above which they are selected
		"""
		self._seq_name = seq_name
		self.vis_threshold = vis_threshold

		self._mot_dir = osp.join(cfg.DATA_DIR, '2DMOT2015')

		self._train_folders = ['Venice-2', 'KITTI-17', 'KITTI-13', 'ADL-Rundle-8', 'ADL-Rundle-6', 'ETH-Pedcross2',
							   'ETH-Sunnyday', 'ETH-Bahnhof', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte']
		self._test_folders = ['Venice-1', 'KITTI-19', 'KITTI-16', 'ADL-Rundle-3', 'ADL-Rundle-1', 'AVG-TownCentre',
							  'ETH-Crossing', 'ETH-Linthescher', 'ETH-Jelmoli', 'PETS09-S2L2', 'TUD-Crossing']

		self.transforms = Compose([ToTensor(), Normalize(normalize_mean, normalize_std)])

		if seq_name:
			assert seq_name in self._train_folders or seq_name in self._test_folders, \
				'Image set does not exist: {}'.format(seq_name)

			self.data = self.sequence(seq_name)
		else:
			self.data = []

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		"""Return the ith image converted to blob"""
		d = self.data[idx]
		# construct image blob and return new dictionary, so blobs are not saved into this class
		im = cv2.imread(d['im_path'])
		blobs, im_scales = test._get_blobs(im)
		data = blobs['data']

		sample = {}
		sample['im_path'] = d['im_path']
		sample['data'] = data
		sample['im_info'] = np.array([data.shape[1], data.shape[2], im_scales[0]], dtype=np.float32)
		# convert to siamese input
		im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		im = Image.fromarray(im)
		im = self.transforms(im)
		sample['app_data'] = im.unsqueeze(0)
		sample['gt'] = {}
		sample['dets'] = []
		# resize tracks
		for k,v in d['gt'].items():
			sample['gt'][k] = v * sample['im_info'][2]
		for det in d['dets']:
			sample['dets'].append(np.hstack([det[:4] * sample['im_info'][2], det[4:5]]))

		return sample

	def __str__(self):
		return self._seq_name

	def sequence(self, seq_name):
		if seq_name in self._train_folders:
			seq_path = osp.join(self._mot_dir, 'train', seq_name)
		else:
			seq_path = osp.join(self._mot_dir, 'test', seq_name)

		im_dir = osp.join(seq_path, 'img1')
		gt_file = osp.join(seq_path, 'gt', 'gt.txt')
		det_file = osp.join(seq_path, 'det', 'det.txt')

		total = []

		boxes = {}
		dets = {}

		valid_files = [f for f in os.listdir(im_dir) if len(re.findall("^[0-9]{6}[.][j][p][g]$", f)) == 1]
		seq_length = len(valid_files)

		for i in range(1, seq_length+1):
			boxes[i] = {}
			dets[i] = []

		if osp.exists(gt_file):
			with open(gt_file, "r") as inf:
				reader = csv.reader(inf, delimiter=',')
				for row in reader:
					#
					if int(row[6]) == 1: #and float(row[8]) >= self.vis_threshold:
						# Make pixel indexes 0-based, should already be 0-based (or not)
						x1 = int(row[2]) - 1
						y1 = int(row[3]) - 1
						# This -1 accounts for the width (width of 1 x1=x2)
						x2 = x1 + float(row[4]) - 1
						y2 = y1 + float(row[5]) - 1
						bb = np.array([x1,y1,x2,y2], dtype=np.float32)
						boxes[int(row[0])][int(row[1])] = bb


		if osp.exists(det_file):
			with open(det_file, "r") as inf:
				reader = csv.reader(inf, delimiter=',')
				for row in reader:
					if len(row) > 0:
						x1 = float(row[2]) - 1
						y1 = float(row[3]) - 1
						# This -1 accounts for the width (width of 1 x1=x2)
						x2 = x1 + float(row[4]) - 1
						y2 = y1 + float(row[5]) - 1
						score = float(row[6])
						bb = np.array([x1,y1,x2,y2, score], dtype=np.float32)
						dets[int(row[0])].append(bb)

		for i in range(1,seq_length+1):
			im_path = osp.join(im_dir,"{:06d}.jpg".format(i))

			sample = { 'gt':boxes[i],
				 	   'im_path':im_path,
					   'dets':dets[i],
			}

			total.append(sample)

		return total

	def write_results(self, all_tracks, output_dir):
		"""Write the tracks in the format for MOT16/MOT17 sumbission

		all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

		Each file contains these lines:
		<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

		Files to sumbit:
		./MOT16-01.txt
		./MOT16-02.txt
		./MOT16-03.txt
		./MOT16-04.txt
		./MOT16-05.txt
		./MOT16-06.txt
		./MOT16-07.txt
		./MOT16-08.txt
		./MOT16-09.txt
		./MOT16-10.txt
		./MOT16-11.txt
		./MOT16-12.txt
		./MOT16-13.txt
		./MOT16-14.txt
		"""

		#format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

		assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		file = osp.join(output_dir, self._seq_name+'.txt')

		print("[*] Writing to: {}".format(file))

		with open(file, "w") as of:
			writer = csv.writer(of, delimiter=',')
			for i, track in all_tracks.items():
				for frame, bb in track.items():
					x1 = bb[0]
					y1 = bb[1]
					x2 = bb[2]
					y2 = bb[3]
					writer.writerow([frame+1, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])

