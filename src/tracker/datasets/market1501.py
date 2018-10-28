import numpy as np
import cv2
import os
import os.path as osp
import configparser
import csv
import h5py
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Normalize, Compose, RandomHorizontalFlip, RandomCrop, ToTensor, RandomResizedCrop

from ..config import cfg


class Market1501(Dataset):
	"""Market1501 dataloader.

	This class builds samples for training of a simaese net. It returns a tripple of 2 matching and 1 not
    matching image crop of a person. The crops con be precalculated.

	Values for P are normally 18 and K 4
	"""

	def __init__(self, seq_name, vis_threshold, P, K, max_per_person, crop_H, crop_W,
				transform, normalize_mean=None, normalize_std=None):

		self.data_dir = osp.join(cfg.DATA_DIR, 'Market-1501-v15.09.15')
		self.seq_name = seq_name

		self.P = P
		self.K = K
		self.max_per_person = max_per_person
		self.crop_H = crop_H
		self.crop_W = crop_W

		if transform == "random":
			self.transform = Compose([RandomCrop((crop_H, crop_W)), RandomHorizontalFlip(), ToTensor(), Normalize(normalize_mean, normalize_std)])
		elif transform == "center":
			self.transform = Compose([CenterCrop((crop_H, crop_W)), ToTensor(), Normalize(normalize_mean, normalize_std)])
		else:
			raise NotImplementedError("Tranformation not understood: {}".format(transform))

		if seq_name:
			assert seq_name in ['bounding_box_test', 'bounding_box_train', 'gt_bbox'], \
				'Image set does not exist: {}'.format(seq_name)
			self.data = self.load_images()
		else:
			self.data = []

		self.build_samples()
		

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		"""Return the ith triplet"""

		res = []
		# idx belongs to the positive sampled person
		pos = self.data[idx]
		res.append(pos[np.random.choice(pos.shape[0], self.K, replace=False)])

		# exclude idx here
		neg_indices = np.random.choice([i for i in range(len(self.data)) if i != idx], self.P-1, replace=False)
		for i in neg_indices:
			neg = self.data[i]
			res.append(neg[np.random.choice(neg.shape[0], self.K, replace=False)])

		# concatenate the results
		r = []
		for pers in res:
			for im in pers:
				im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
				im = Image.fromarray(im)
				r.append(self.transform(im))
		images = torch.stack(r, 0)

		# construct the labels
		labels = [idx] * self.K

		for l in neg_indices:
			labels += [l] * self.K

		labels = np.array(labels)

		batch = [images, labels]

		return batch

	def load_images(self):
		im_folder = osp.join(self.data_dir, self.seq_name)

		total = []

		for f in os.listdir(im_folder):
			if os.path.isfile(os.path.join(im_folder, f)) and f[-4:] == ".jpg":
				im_path = osp.join(im_folder,f)
				sample = {'im_path':im_path,
					  	  'id':int(f[:4])}

				total.append(sample)

		return total

	def build_samples(self):
		"""Builds the samples for simaese out of the data."""

		tracks = {}

		for sample in self.data:
			im_path = sample['im_path']
			identity = sample['id']

			if identity in tracks:
				tracks[identity].append(sample)
			else:
				tracks[identity] = []
				tracks[identity].append(sample)

		# sample max_per_person images and filter out tracks smaller than 4 samples
		#outdir = get_output_dir("siamese_test")
		res = []
		for k,v in tracks.items():
			l = len(v)
			if l >= self.K:
				pers = []
				if l > self.max_per_person:
					for i in np.random.choice(l, self.max_per_person, replace=False):
						pers.append(self.build_crop(v[i]['im_path']))
				else:
					for i in range(l):
						pers.append(self.build_crop(v[i]['im_path']))
				res.append(np.array(pers))

		if self.seq_name:
			print("[*] Loaded {} persons from sequence {}.".format(len(res), self.seq_name))

		self.data = res

	def build_crop(self, im_path):
		im = cv2.imread(im_path)

		im = cv2.resize(im, (int(self.crop_W*1.125), int(self.crop_H*1.125)), interpolation=cv2.INTER_LINEAR)

		return im