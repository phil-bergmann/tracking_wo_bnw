
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

		return db_inds

	@staticmethod
	def _to_blobs(minibatch_db):
		for smp in minibatch_db:
			smp['blobs'] = []
			for pth in smp['im_paths']:
				im = cv2.imread(pth)
				blobs, im_scales = _get_blobs(im)
				im_blob = blobs['data']
				blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
				smp['blobs'].append(blobs)


	def _get_next_minibatch(self):
		"""Return next minibatch

		At the moment only batches of size 1
		"""
		db_inds = self._get_next_minibatch_inds()
		minibatch_db = [self._db[i] for i in db_inds]
		DataLayer._to_blobs(minibatch_db)
		return minibatch_db


	def forward(self):
		"""Return the blobs

		minibatch is a list containing dictionaries d with keys ['tracks', 'im_paths', 'blobs']
		d['im_paths']: Paths to the 3 images
		d['tracks']: Tracks through the 3 images
		d['blobs']: Another list containing the 3 blob dictionaries
		d['blobs'][i]: Dictionary containing the blob with keys ['data', 'im_info']

		blobs = [ {
				'tracks': numpy array ((num_tracks,3,4)),
				'im_paths': [im_path1, im_path2, im_path3]
				'blobs': [3 * {'data':d, 'im_info':info},]
		} ]
		"""
		minibatch = self._get_next_minibatch()
		return minibatch[0]