
from model.test import _get_blobs

from .mot_sequence import MOT_Sequence

import cv2
import numpy as np


class MOT_Tracks(MOT_Sequence):
	"""Multiple Object Tracking Dataset.

	This class builds tracks out of a sequence. First tracks are constructed with all from the
	timestep where the person first appears until the end. When the person is not visible this
	frame is marked as inactive. Then these tracks are splitted into shorter ones. They have
	to begin with a active frame and can end with a maximum of 3 inactive frames.
	"""

	def __init__(self, seq_name=None):
		super().__init__(seq_name)

		self.build_tracks()

	def __getitem__(self, idx):
		"""Return the ith track with images converted to blobs"""
		track = self.data[idx]
		res = []
		# construct image blobs and return new list, so blobs are not saved into this class
		for f in track:
			im = cv2.imread(f['im_path'])
			blobs, im_scales = _get_blobs(im)
			data = blobs['data']

			sample = {}
			sample['id'] = f['id']
			sample['im_path'] = f['im_path']
			sample['data'] = data
			sample['im_info'] = np.array([data.shape[1], data.shape[2], im_scales[0]], dtype=np.float32)
			if 'gt' in f.keys():
				sample['gt'] = f['gt'] * sample['im_info'][2]
			sample['active'] = f['active']

			res.append(sample)

		return res

	def build_tracks(self):
		"""Builds the tracks out of the sequence

		Format for tracks is a list [] containing dictionaries with the keys id, im_path, gt
		id is the id of the track if later the disconnected tracks from occlusions want to be
		reconnected again and gt is the bb of the person in this frame.

		Sample small tracks <= 7
		"""

		tracks = {}

		for sample in self.data:
			im_path = sample['im_path']
			gt = sample['gt']

			for k,v in tracks.items():
				if k in gt.keys():
					v.append({'id':k, 'im_path':im_path, 'gt':gt[k], 'active':True})
					del gt[k]
				else:
					v.append({'id':k, 'im_path':im_path, 'active':False})

			# For all remaining BB in gt new tracks are created
			for k,v in gt.items():
				tracks[k] = [{'id':k, 'im_path':im_path, 'gt':v, 'active':True}]

		# Now begin to split into subtracks
		res = []
		for _,track in tracks.items():
			t = []
			for v in track:
				if v['active']:
					# if sequence True ... False and new object True we finish it and create new one
					if len(t) > 0 and t[-1]['active'] == False:
						res.append(t)
						t = []
					t.append(v)
				# no inactive samples at beginning of sequence
				elif len(t) > 0:
					t.append(v)
				# track finished if too long or 3 times inactive at end
				if (len(t) >= 7) or (len(t) >= 4 and t[-3]['active'] == False):
					res.append(t)
					t = []


		if self._seq_name:
			print("[*] Loaded {} tracks from sequence {}.".format(len(res), self._seq_name))

		self.data = res
