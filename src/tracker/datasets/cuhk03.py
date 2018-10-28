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


class CUHK03(Dataset):
	"""CUHK03 dataloader.

	Inspired from https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data_manager/cuhk03.py.

	This class builds samples for training of a simaese net. It returns a tripple of 2 matching and 1 not
    matching image crop of a person. The crops con be precalculated.

	Values for P are normally 18 and K 4
	"""

	def __init__(self, seq_name, vis_threshold, P, K, max_per_person, crop_H, crop_W,
				transform, normalize_mean=None, normalize_std=None):

		self.data_dir = osp.join(cfg.DATA_DIR, 'cuhk03_release')
		self.raw_mat_path = osp.join(self.data_dir, 'cuhk-03.mat')
		if not osp.exists(self.raw_mat_path):
			raise RuntimeError("'{}' is not available".format(self.raw_mat_path))
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
			assert seq_name in ['labeled', 'detected']
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
		"""Loads the images from the mat file and saves them."""

		#save_path = os.path.join(self.data_dir, self.seq_name)
		#if not osp.exists(save_path):
		#	os.makedirs(save_path)

		mat = h5py.File(self.raw_mat_path, 'r')

		identities = {} # maps entries of the file to identity numbers

		total = []

		def _deref(ref):
			"""Matlab reverses the order of column / row."""
			return mat[ref][:].T

		for campid, camp_ref in enumerate(mat[self.seq_name][0]):
			camp = _deref(camp_ref)     # returns the camera pair
			num_pids = camp.shape[0]    # gets number of identities
			for pid in range(num_pids):
				img_paths = []
				for imgid, img_ref in enumerate(camp[pid,:]):
					img = _deref(img_ref)                      # now we have a single image
					if img.size == 0 or img.ndim < 3: continue # if empty skip
					img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # switch to BGR from RGB

					id_name = '{:01d}_{:03d}'.format(campid+1, pid+1)
					if id_name in identities:
						id_num = identities[id_name]
					else:
						id_num = len(identities)
						identities[id_name] = id_num

					#im_name = '{:04d}_{:01d}.png'.format(id_num, imgid)
					#im_path = os.path.join(save_path, im_name)
					#if not os.path.isfile(im_path):
					#	cv2.imwrite(im_path, img)
					sample = {'im':self.build_crop(img),
							  'id':id_num}
					total.append(sample)

		return total

	def build_samples(self):
		"""Builds the samples for simaese out of the data."""

		tracks = {}

		for sample in self.data:
			im = sample['im']
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
						pers.append(v[i]['im'])
				else:
					for i in range(l):
						pers.append(v[i]['im'])
				res.append(np.array(pers))

		if self.seq_name:
			print("[*] Loaded {} persons from sequence {}.".format(len(res), self.seq_name))

		self.data = res

	def build_crop(self, im):

		im = cv2.resize(im, (int(self.crop_W*1.125), int(self.crop_H*1.125)), interpolation=cv2.INTER_LINEAR)

		return im