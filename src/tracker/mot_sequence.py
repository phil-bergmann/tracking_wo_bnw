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


class MOT_Sequence(Dataset):
	"""Multiple Object Tracking Dataset.

	This dataloader is designed so that it can handle only one sequence, if more have to be
	handled one should inherit from this class.
	"""

	def __init__(self, seq_name=None, vis_threshold=0.0, normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225]):
		"""
		Args:
			seq_name (string): Sequence to take
			vis_threshold (float): Threshold of visibility of persons above which they are selected
		"""
		self._seq_name = seq_name
		self.vis_threshold = vis_threshold

		self._mot_dir = osp.join(cfg.DATA_DIR, 'MOT17Det')
		self._label_dir = osp.join(cfg.DATA_DIR, 'MOT16Labels')
		self._raw_label_dir = osp.join(cfg.DATA_DIR, 'MOT16-det-dpm-raw')

		self._train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
			'MOT17-11', 'MOT17-13']
		self._test_folders = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07',
			'MOT17-08', 'MOT17-12', 'MOT17-14']

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
		sample['vis'] = {}
		sample['dets'] = []
		sample['raw_dets'] = []
		# resize tracks
		for k,v in d['gt'].items():
			sample['gt'][k] = v * sample['im_info'][2]
		for k,v in d['vis'].items():
			sample['vis'][k] = v
		for det in d['dets']:
			sample['dets'].append(det * sample['im_info'][2])
		for raw_det in d['raw_dets']:
			sample['raw_dets'].append(raw_det * sample['im_info'][2])

		return sample

	def sequence(self, seq_name):
		if seq_name in self._train_folders:
			seq_path = osp.join(self._mot_dir, 'train', seq_name)
			label_path = osp.join(self._label_dir, 'train', 'MOT16-'+seq_name[-2:])
		else:
			seq_path = osp.join(self._mot_dir, 'test', seq_name)
			label_path = osp.join(self._label_dir, 'test', 'MOT16-'+seq_name[-2:])
		raw_label_path = osp.join(self._raw_label_dir, 'MOT16-'+seq_name[-2:])
			
		config_file = osp.join(seq_path, 'seqinfo.ini')

		assert osp.exists(config_file), \
			'Config file does not exist: {}'.format(config_file)

		config = configparser.ConfigParser()
		config.read(config_file)
		seqLength = int(config['Sequence']['seqLength'])
		imWidth = int(config['Sequence']['imWidth'])
		imHeight = int(config['Sequence']['imHeight'])
		imExt = config['Sequence']['imExt']
		imDir = config['Sequence']['imDir']

		imDir = osp.join(seq_path, imDir)
		gt_file = osp.join(seq_path, 'gt', 'gt.txt')
		det_file = osp.join(label_path, 'det', 'det.txt')
		raw_det_file = osp.join(raw_label_path, 'det', 'det-dpm-raw.txt')

		total = []
		train = []
		val = []

		visibility = {}
		boxes = {}
		dets = {}
		raw_dets = {}

		for i in range(1, seqLength+1):
			boxes[i] = {}
			visibility[i] = {}
			dets[i] = []
			raw_dets[i] = []

		if osp.exists(gt_file):
			with open(gt_file, "r") as inf:
				reader = csv.reader(inf, delimiter=',')
				for row in reader:
					# class person, certainity 1, visibility >= 0.25
					if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= self.vis_threshold:
						# Make pixel indexes 0-based, should already be 0-based (or not)
						x1 = int(row[2]) - 1
						y1 = int(row[3]) - 1
						# This -1 accounts for the width (width of 1 x1=x2)
						x2 = x1 + int(row[4]) - 1
						y2 = y1 + int(row[5]) - 1
						bb = np.array([x1,y1,x2,y2], dtype=np.float32)
						boxes[int(row[0])][int(row[1])] = bb
						visibility[int(row[0])][int(row[1])] = float(row[8])


		if osp.exists(det_file):
			with open(det_file, "r") as inf:
				reader = csv.reader(inf, delimiter=',')
				for row in reader:
					x1 = float(row[2]) - 1
					y1 = float(row[3]) - 1
					# This -1 accounts for the width (width of 1 x1=x2)
					x2 = x1 + float(row[4]) - 1
					y2 = y1 + float(row[5]) - 1
					bb = np.array([x1,y1,x2,y2], dtype=np.float32)
					dets[int(row[0])].append(bb)

		if osp.exists(raw_det_file):
			with open(raw_det_file, "r") as inf:
				reader = csv.reader(inf, delimiter=',')
				for row in reader:
					x1 = float(row[2]) - 1
					y1 = float(row[3]) - 1
					# This -1 accounts for the width (width of 1 x1=x2)
					x2 = x1 + float(row[4]) - 1
					y2 = y1 + float(row[5]) - 1
					bb = np.array([x1,y1,x2,y2], dtype=np.float32)
					raw_dets[int(row[0])].append(bb)


		for i in range(1,seqLength+1):
			im_path = osp.join(imDir,"{:06d}.jpg".format(i))

			sample = { 'gt':boxes[i],
				 	   'im_path':im_path,
				 	   'vis':visibility[i],
					   'dets':dets[i],
					   'raw_dets':raw_dets[i],
			}

			total.append(sample)
			if i <= seqLength*0.5:
				train.append(sample)
			if i >= seqLength*0.75:
				val.append(sample)

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

		file = osp.join(output_dir, 'MOT16-'+self._seq_name[6:8]+'.txt')

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

