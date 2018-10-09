import numpy as np
import cv2
import os
import os.path as osp
import configparser
import csv
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Compose, ToTensor

from model.test import _get_blobs

from .config import cfg


class KITTI_Sequence(Dataset):
	"""KITTI Tracking Dataset.

	This dataloader is designed so that it can handle only one sequence, if more have to be
	handled one should inherit from this class.
	"""

	def __init__(self, seq_name=None, vis_threshold=[2, 3], normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225]):
		"""
		Args:
			seq_name (string): Sequence to take (e.g. "train_0005_Car", "test_0026_Pedestrian")
			vis_threshold (float): Threshold of visibility of persons above which they are selected
		"""
		self._seq_name = seq_name
		self._trunc_threshold = vis_threshold[0]
		self._occl_threshold = vis_threshold[1]

		self._kitti_dir = osp.join(cfg.DATA_DIR, 'KITTI_tracking')
		#self._label_dir = osp.join(cfg.DATA_DIR, 'MOT16Labels')

		self._train_folders = ["%04d"%(seq) for seq in range(21)]
		self._test_folders = ["%04d"%(seq) for seq in range(29)]

		self.transforms = Compose([ToTensor(), Normalize(normalize_mean, normalize_std)])

		if seq_name:
			self._tt = seq_name.split("_")[0]
			self._seq_num = seq_name.split("_")[1]
			self._cl = seq_name.split("_")[2]
			assert self._cl in ["Car", "Pedestrian"], \
				'Image set does not exist: {}'.format(seq_name)
			if "train" == self._tt:
				assert self._seq_num in self._train_folders, \
					'Image set does not exist: {}'.format(seq_name)
			elif "test" == self._tt:
				assert self._seq_num in self._test_folders, \
					'Image set does not exist: {}'.format(seq_name)
			else:
				assert False, 'Image set does not exist: {}'.format(seq_name)
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
		blobs, im_scales = _get_blobs(im)
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
		sample['occl'] = {}
		sample['trunc'] = {}
		# resize tracks
		for k,v in d['gt'].items():
			sample['gt'][k] = v * sample['im_info'][2]
		for k,v in d['occl'].items():
			sample['occl'][k] = v
		for k,v in d['trunc'].items():
			sample['trunc'][k] = v

		return sample

	def sequence(self, seq_name):
		if "train" == self._tt:
			im_dir = osp.join(self._kitti_dir, 'training', 'image_02', self._seq_num)
			gt_file = osp.join(self._kitti_dir, 'training', 'label_02', self._seq_num+".txt")
		else:
			im_dir = osp.join(self._kitti_dir, 'testing', 'image_02', self._seq_num)
			gt_file = None

			
		valid_names = ["%06d.png"%i for i in range(1000000)]
		seqLength = len([i for i in os.listdir(im_dir) if i in valid_names])

		total = []

		occlusion = {}
		boxes = {}
		truncation = {}

		for i in range(0, seqLength):
			boxes[i] = {}
			truncation[i] = {}
			occlusion[i] = {}

		if gt_file and osp.exists(gt_file):
			with open(gt_file, "r") as inf:
				reader = csv.reader(inf, delimiter=' ')
				for row in reader:
					# class person, certainity 1, visibility >= 0.25
					if row[2] == self._cl and int(row[3]) <= self._trunc_threshold and int(row[4]) <= self._occl_threshold:
						# Make pixel indexes 0-based, should already be 0-based (or not)
						x1 = int(float(row[6]))
						y1 = int(float(row[7]))
						# This -1 accounts for the width (width of 1 x1=x2)
						x2 = int(float(row[8]))
						y2 = int(float(row[9]))
						bb = np.array([x1,y1,x2,y2], dtype=np.float32)
						boxes[int(row[0])][int(row[1])] = bb
						truncation[int(row[0])][int(row[1])] = int(row[3])
						occlusion[int(row[0])][int(row[1])] = int(row[4])
						#print(row)

		for i in range(0, seqLength):
			im_path = osp.join(im_dir,"{:06d}.png".format(i))

			sample = { 'gt':boxes[i],
					   'im_path':im_path,
					   'trunc':truncation[i],
					   'occl':occlusion[i],
			}

			total.append(sample)

		return total

	def write_results(self, all_tracks, output_dir):
		"""Write the tracks in the format for KITTI tracking submission

		all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

		"""

		assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		file = osp.join(output_dir, self._seq_num+'.txt')

		print("[*] Writing to: {}".format(file))

		with open(file, "w") as of:
			writer = csv.writer(of, delimiter=' ')
			for i, track in all_tracks.items():
				for frame, bb in track.items():
					x1 = bb[0]
					y1 = bb[1]
					x2 = bb[2]
					y2 = bb[3]
					score = bb[4]
					writer.writerow([frame, i, self._cl, 0, 0, 0, x1, y1, x2, y2, 0, 0, 0, 0, 0, 0, 0, score])
