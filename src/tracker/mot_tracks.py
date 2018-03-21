
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

	def __init__(self, seq_name, track_len, mult_transitions, max_trail_dead, sample_transitions_backwards,
		min_alive_start, active_thresh, generate_blobs=True):
		super().__init__(seq_name)
		self.track_len = track_len
		self.mult_transitions = mult_transitions
		self.max_trail_dead = max_trail_dead
		self.sample_transitions_backwards = sample_transitions_backwards
		self.min_alive_start = min_alive_start
		self.active_thresh = active_thresh


		self.build_tracks()

		self.generate_blobs=generate_blobs

	def __getitem__(self, idx):
		"""Return the ith track with images converted to blobs"""
		track = self.data[idx]
		res = []
		# construct image blobs and return new list, so blobs are not saved into this class
		for f in track:
			sample = {}
			if self.generate_blobs:
				im = cv2.imread(f['im_path'])
				blobs, im_scales = _get_blobs(im)
				data = blobs['data']
				sample['data'] = data
				sample['im_info'] = np.array([data.shape[1], data.shape[2], im_scales[0]], dtype=np.float32)
				if 'gt' in f.keys():
					sample['gt'] = f['gt'] * sample['im_info'][2]
			else:
				if 'gt' in f.keys():
					sample['gt'] = f['gt']
			
			sample['id'] = f['id']
			sample['im_path'] = f['im_path']
			sample['active'] = f['active']
			sample['vis'] = f['vis']

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
			vis = sample['vis']

			for k,v in tracks.items():
				if k in gt.keys():
					active = vis[k] >= self.active_thresh
					v.append({'id':k, 'im_path':im_path, 'gt':gt[k], 'active':active, 'vis':vis[k]})
					del gt[k]

			# For all remaining BB in gt new tracks are created
			for k,v in gt.items():
				if vis[k] >= self.active_thresh:
					tracks[k] = [{'id':k, 'im_path':im_path, 'gt':v, 'active':True, 'vis':vis[k]}]

		# Now begin to split into subtracks
		res, weights = self.split_tracks(tracks)

		# duplicate tracks (go in both directions)
		if self.sample_transitions_backwards:
			tracks_back = {}
			for k,v in tracks.items():
				t0 = tracks[k]
				t1 = []
				for j in range(len(t0)-1,-1,-1):
					t1.append(t0[j])
				tracks_back[k+1000] = t1
			r, w = self.split_tracks(tracks_back, True)
			res += r
			weights += w

		if self._seq_name:
			print("[*] Loaded {} tracks from sequence {}.".format(len(res), self._seq_name))

		# calculate weights
		"""
		weights = []
		for track in res:
			vis = []
			for f in track:
				vis.append(f['vis'])
			#weights.append(1.05-np.mean(vis)**0.3)
			if np.mean(vis) < 0.9:
				weights.append(5)
			else:
				weights.append(1)
		"""
		
		self.weights = weights

		self.data = res

	def split_tracks(self, tracks, only_trans=False):
		res = []
		weights = []
		for _,track in tracks.items():
			t = []
			for v in track:
				if v['active']:
					# if sequence True ... False and new object True we finish it and create new one
					if len(t) > 0 and t[-1]['active'] == False:
						#for i in range(dupl_transitions):
						res.append(t)
						weights.append(self.mult_transitions)
						t = []
					t.append(v)
				# no inactive samples at beginning of sequence
				elif len(t) >= self.min_alive_start:
					t.append(v)
				# not enough alive at beginning
				else:
					t = []
				# track finished if too long
				if (len(t) >= self.track_len) or (len(t) >= (self.max_trail_dead+1) and t[-self.max_trail_dead]['active'] == False):
				#if len(t) >= length:
					if t[-1]['active'] == False:
						#for i in range(dupl_transitions):
						res.append(t)
						weights.append(self.mult_transitions)
					elif only_trans == False:
						res.append(t)
						weights.append(1)
					t = []
		return res, weights