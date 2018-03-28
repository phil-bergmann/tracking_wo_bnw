
from model.test import _get_blobs

from .mot_sequence import MOT_Sequence
#from .config import get_output_dir

import cv2
import numpy as np
import os.path as osp


class MOT_Siamese(MOT_Sequence):
	"""Multiple Object Tracking Dataset.

	This class builds samples for training of a simaese net. It returns a tripple of 2 matching and 1 not
    matching image crop of a person. The crops con be precalculated.
	"""

	def __init__(self, seq_name=None, vis_threshold=0.5, P=18, K=4, max_per_person=40, crop_H=224, crop_W=112):
		super().__init__(seq_name, vis_threshold=vis_threshold)

		self.P = P
		self.K = K
		self.max_per_person = max_per_person
		self.crop_H = crop_H
		self.crop_W = crop_W

		self.build_samples()

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
		res = np.concatenate(res, axis=0)

		# construct the labels
		labels = [idx] * self.K

		for l in neg_indices:
			labels += [l] * self.K

		labels = np.array(labels)

		batch = [res, labels]

		return batch

	def build_samples(self):
		"""Builds the samples out of the sequence
		"""

		tracks = {}

		for sample in self.data:
			im_path = sample['im_path']
			gt = sample['gt']
			vis = sample['vis']

			for k,v in tracks.items():
				if k in gt.keys():
					v.append({'id':k, 'im_path':im_path, 'gt':gt[k], 'vis':vis[k]})
					del gt[k]

			# For all remaining BB in gt new tracks are created
			for k,v in gt.items():
				tracks[k] = [{'id':k, 'im_path':im_path, 'gt':v, 'vis':vis[k]}]

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
		blobs, im_scales = _get_blobs(im)
		im = blobs['data'][0]
		gt = gt * im_scales[0]
		gt = np.clip(gt, 0, None)

		im = im[int(gt[1]):int(gt[3]), int(gt[0]):int(gt[2])]

		im = cv2.resize(im, (self.crop_W, self.crop_H), interpolation=cv2.INTER_LINEAR)

		return im
