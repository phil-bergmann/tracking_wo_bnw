# from model.test import _get_blobs

from .mot_sequence import MOT17Sequence
from ..config import get_output_dir

import cv2
from PIL import Image
import numpy as np
import os.path as osp

import torch
from torchvision.transforms import CenterCrop, Normalize, Compose, RandomHorizontalFlip, RandomCrop, ToTensor, RandomResizedCrop


class MOTreID(MOT17Sequence):
	"""Multiple Object Tracking Dataset.

	This class builds samples for training of a simaese net. It returns a tripple of 2 matching and 1 not
    matching image crop of a person. The crops con be precalculated.

	Values for P are normally 18 and K 4
	"""

	def __init__(self, seq_name, split, vis_threshold, P, K, max_per_person, crop_H, crop_W,
				transform, normalize_mean=None, normalize_std=None):
		super().__init__(seq_name, vis_threshold=vis_threshold)

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

		self.build_samples()

		if split == 'train':
			pass
		elif split == 'small_train':
			self.data = self.data[0::5] + self.data[1::5] + self.data[2::5] + self.data[3::5]
		elif split == 'small_val':
			self.data = self.data[4::5]
		else:
			raise NotImplementedError("Split: {}".format(split))

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

	def build_samples(self):
		"""Builds the samples out of the sequence."""

		tracks = {}

		for sample in self.data:
			im_path = sample['im_path']
			gt = sample['gt']

			for k,v in tracks.items():
				if k in gt.keys():
					v.append({'id':k, 'im_path':im_path, 'gt':gt[k]})
					del gt[k]

			# For all remaining BB in gt new tracks are created
			for k,v in gt.items():
				tracks[k] = [{'id':k, 'im_path':im_path, 'gt':v}]

		# sample max_per_person images and filter out tracks smaller than 4 samples
		#outdir = get_output_dir("siamese_test")
		res = []
		for k,v in tracks.items():
			l = len(v)
			if l >= self.K:
				pers = []
				if l > self.max_per_person:
					for i in np.random.choice(l, self.max_per_person, replace=False):
						pers.append(self.build_crop(v[i]['im_path'], v[i]['gt']))
				else:
					for i in range(l):
						pers.append(self.build_crop(v[i]['im_path'], v[i]['gt']))

				#for i,v in enumerate(pers):
				#	cv2.imwrite(osp.join(outdir, str(k)+'_'+str(i)+'.png'),v)
				res.append(np.array(pers))

		if self._seq_name:
			print("[*] Loaded {} persons from sequence {}.".format(len(res), self._seq_name))

		self.data = res

	def build_crop(self, im_path, gt):
		im = cv2.imread(im_path)
		height, width, channels = im.shape
		#blobs, im_scales = _get_blobs(im)
		#im = blobs['data'][0]
		#gt = gt * im_scales[0]
		# clip to image boundary
		w = gt[2] - gt[0]
		h = gt[3] - gt[1]
		context = 0
		gt[0] = np.clip(gt[0]-context*w, 0, width-1)
		gt[1] = np.clip(gt[1]-context*h, 0, height-1)
		gt[2] = np.clip(gt[2]+context*w, 0, width-1)
		gt[3] = np.clip(gt[3]+context*h, 0, height-1)

		im = im[int(gt[1]):int(gt[3]), int(gt[0]):int(gt[2])]

		im = cv2.resize(im, (int(self.crop_W*1.125), int(self.crop_H*1.125)), interpolation=cv2.INTER_LINEAR)

		return im
