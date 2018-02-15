
from model.test import _get_blobs

from .mot_sequence import MOT_Sequence

import cv2
import numpy as np


class MOT_Tracks(MOT_Sequence):
	"""Multiple Object Tracking Dataset.

	This class builds tracks out of a sequence. At the moment when a person disappears and
	reappears a new track should be created.
	"""

	def __init__(self, seq_name=None):
		super().__init__(seq_name)

		self.build_tracks()

	def __getitem__(self, idx):
		"""Return the ith track with images converted to blobs"""
		d = self.data[idx]
		track = []
		# construct image blobs and return new list, so blobs are not saved into this class
		for t in d:
			im = cv2.imread(t['im_path'])
			blobs, im_scales = _get_blobs(im)
			data = blobs['data']

			sample = {}
			sample['id'] = t['id']
			sample['im_path'] = t['im_path']
			sample['data'] = data
			sample['im_info'] = np.array([data.shape[1], data.shape[2], im_scales[0]], dtype=np.float32)
			sample['gt'] = t['gt'] * sample['im_info'][2]

			track.append(sample)

		return track

	def build_tracks(self):
		"""Builds the tracks out of the sequence

		Format for tracks is a list [] containing dictionaries with the keys id, im_path, gt
		id is the id of the track if later the disconnected tracks from occlusions want to be
		reconnected again and gt is the bb of the person in this frame.

		Sample small tracks <= 7

		Also filter out small track lengths < 2. (That are no real tracks)
		"""

		tracks = []
		tmp = []

		for sample in self.data:
			im_path = sample['im_path']
			gt = sample['gt']

			# check if for samples in tmp there exist BB in next frame
			# If yes append BB and delete from gt, else track is finished
			# !!Take care while deleting elements in the list you are iterating, list() makes copy
			for t in list(tmp):
				idx = t[0]['id']
				if idx in gt.keys() and len(t) < 7:
					t.append({'id':idx, 'im_path':im_path, 'gt':gt[idx]})
					del gt[idx]
				else:
					tracks.append(t)
					tmp.remove(t)

			# For all remaining BB in gt new tracks are created
			for k,v in gt.items():
				t = []
				t.append({'id':k, 'im_path':im_path, 'gt':v})
				tmp.append(t)

		# now filter the tracks if they are shorter than 2 frames
		# !!Take care while deleting elements in the list you are iterating, list() makes copy
		num_tracks = len(tracks)
		for t in list(tracks):
			if len(t) < 2:
				tracks.remove(t)

		if self._seq_name:
			print("[*] Filtered out {}/{} tracks in sequence {}.".format(num_tracks-len(tracks), num_tracks, self._seq_name))

		self.data = tracks
