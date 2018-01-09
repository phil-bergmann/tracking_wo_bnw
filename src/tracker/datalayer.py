
import numpy as np
import time
import cv2

from model.test import _get_blobs

from .config import cfg

class DataLayer(object):

	def __init__(self, db, val=False):
		self._db = db
		self._val = val
		self._shuffle_db_inds()
		#print("[*] Converting to blobs...")
		#DataLayer._to_blobs(self._db)
		#print("[*] Finished!")


	def _shuffle_db_inds(self):
		if self._val:
			st0 = np.random.get_state()
			millis = int(round(time.time() * 1000)) % 4294967295
			np.random.seed(millis)

		self._perm = np.random.permutation(np.arange(len(self._db)))

		if self._val:
			np.random.set_state(st0)

		self._cur = 0

	def _get_next_minibatch_inds(self):
		"""Return the roidb indices for the next minibatch."""
	
		if self._cur + cfg.TRAIN.SMP_PER_BATCH >= len(self._db):
			self._shuffle_db_inds()

		db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.SMP_PER_BATCH]
		self._cur += cfg.TRAIN.SMP_PER_BATCH

		#print("{}/{}".format(db_inds,len(self._db)))

		return db_inds

	# Wrong as memory all blobs are hold in memory this way and ram usage goes big
	@staticmethod
	def _to_blobs_old(minibatch_db):
		for smp in minibatch_db:
			# for performance check if blobs are already there
			if 'blobs' not in smp.keys():
				smp['blobs'] = {}
				smp['blobs']['data'] = []
				for i,pth in enumerate(smp['im_paths']):
					im = cv2.imread(pth)
					blobs, im_scales = _get_blobs(im)
					im_blob = blobs['data']
					if i == 0:
						smp['blobs']['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
					smp['blobs']['data'].append(im_blob)
				# Also add resized tracks to blobs
				smp['blobs']['tracks'] = smp['tracks'] * smp['blobs']['im_info'][0,2]


	@staticmethod
	def _to_blobs(minibatch_db):
		blobs = []
		for smp in minibatch_db:
			blob = {}
			blob['data'] = []
			blob['im_paths'] = []
			for i,pth in enumerate(smp['im_paths']):
				im = cv2.imread(pth)
				data, im_scales = _get_blobs(im)
				data = data['data']
				if i == 0:
					blob['im_info'] = np.array([[data.shape[1], data.shape[2], im_scales[0]]], dtype=np.float32)
				blob['data'].append(data)
				# Also add resized tracks to blobs
				if 'tracks' in smp.keys():
					blob['tracks'] = smp['tracks'] * blob['im_info'][0,2]
				blob['im_paths'].append(pth)
			blobs.append(blob)

		return blobs


	def _get_next_minibatch(self):
		"""Return next minibatch

		At the moment only batches of size 1
		"""
		db_inds = self._get_next_minibatch_inds()
		minibatch_db = [self._db[i] for i in db_inds]
		# Perhaps don't do that every time?
		blobs = DataLayer._to_blobs(minibatch_db)
		return blobs

	def _get_next_precalc_minibatch(self):
		db_inds = self._get_next_minibatch_inds()
		minibatch_db = [self._db[i] for i in db_inds]
		return minibatch_db[0]

	def forward(self):
		"""Return the blobs

		minibatch is a list containing dictionaries d with keys ['tracks', 'im_paths', 'blobs']
		d['im_paths']: Paths to the 2 images
		d['tracks']: Tracks through the 2 images
		d['blobs']: Another list containing the 2 blob dictionaries
		d['blobs']['im_info'][0]: [im_blob.shape[1], im_blob.shape[2], im_scales[0]]
		d['blobs']['data'][i]: blob data

		blobs = [ {
				'tracks': numpy array ((num_tracks,3,4)),
				'im_paths': [im_path1, im_path2, im_path3]
				'blobs': [3 * {'data':d, 'im_info':info},]
		} ]
		"""
		blobs = self._get_next_minibatch()
		return blobs[0]