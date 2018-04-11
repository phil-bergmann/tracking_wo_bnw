from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class Simple_SiameseID_Tracker():
	"""
	This tracker uses the siamese appearance features to decide wether a track hast to die or not (without nms)
	"""

	def __init__(self, frcnn, cnn, detection_person_thresh, regression_person_thresh, detection_nms_thresh,
		regression_nms_thresh, use_detections):
		self.frcnn = frcnn
		self.cnn = cnn
		self.detection_person_thresh = detection_person_thresh
		self.regression_person_thresh = regression_person_thresh
		self.detection_nms_thresh = detection_nms_thresh
		self.regression_nms_thresh = regression_nms_thresh
		self.use_detections = use_detections
		self.id_threshold = 0.3
		self.inactive_patience = 10
		self.do_reid = False

		self.reset()

	def reset(self, hard=True):
		self.tracks = []
		self.inactive_tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def keep(self, keep):
		tracks = []
		for i in keep:
			tracks.append(self.tracks[i])
		self.inactive_tracks += [t for t in self.tracks if t not in tracks]
		self.tracks = tracks

	def add(self, new_det_pos, new_det_features):
		num_new = new_det_pos.size(0)
		for i in range(num_new):
			self.tracks.append(Track(new_det_pos[i].view(1,-1), self.track_num + i, new_det_features[i].view(1,-1),
										self.regression_person_thresh, self.inactive_patience))
		self.track_num += num_new

	def regress_tracks(self, blob):
		cl = 1

		pos = self.get_pos()

		# regress
		_, _, bbox_pred, rois = self.frcnn.test_rois(pos)
		boxes = bbox_transform_inv(rois, bbox_pred)
		boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data
		pos = boxes[:,cl*4:(cl+1)*4]
		for t,p in zip(self.tracks, pos):
			t.pos = p.view(1,-1)

		# get scores of new regressed positions
		_, scores, _, _ = self.frcnn.test_rois(pos)
		scores = scores[:,cl]

		for i in range(len(self.tracks)-1,-1,-1):
			t = self.tracks[i]
			if t.is_dead(scores[i]):
				self.tracks.remove(t)
				self.inactive_tracks.append(t)

	def get_pos(self):
		if len(self.tracks) == 1:
			pos = self.tracks[0].pos
		elif len(self.tracks) > 1:
			pos = torch.cat([t.pos for t in self.tracks],0)
		else:
			pos = torch.zeros(0).cuda()
		return pos

	def get_features(self):
		if len(self.tracks) == 1:
			features = self.tracks[0].features
		elif len(self.tracks) > 1:
			features = torch.cat([t.features for t in self.tracks],0)
		else:
			features = torch.zeros(0).cuda()
		return features
	
	def get_inactive_features(self):
		if len(self.inactive_tracks) == 1:
			features = self.inactive_tracks[0].features
		elif len(self.inactive_tracks) > 1:
			features = torch.cat([t.features for t in self.inactive_tracks],0)
		else:
			features = torch.zeros(0).cuda()
		return features

	def reid(self, blob, new_det_pos):
		new_det_features = self.cnn.test_rois(blob['data'][0], new_det_pos).data
		if len(self.inactive_tracks) >= 1 and self.do_reid:
			inactive_features = self.get_inactive_features()

			dist_mat = cdist(inactive_features.cpu().numpy(), new_det_features.cpu().numpy(), 'cosine')

			row_ind, col_ind = linear_sum_assignment(dist_mat)

			assigned = []
			remove_inactive = []
			for r,c in zip(row_ind, col_ind):
				if dist_mat[r,c] <= self.reid_threshold:
					#print("im: {}\ncosts: {}".format(self.im_index, costs[r,c]))
					t = self.inactive_tracks[r]
					self.tracks.append(t)
					t.count_dead = 0
					t.pos = new_det_pos[c].view(1,-1)
					t.features = new_det_features[c].view(1,-1)
					assigned.append(c)
					remove_inactive.append(t)

			for t in remove_inactive:
				self.inactive_tracks.remove(t)

			keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
			if keep.nelement() > 0:
				new_det_pos = new_det_pos[keep]
				new_det_features = new_det_features[keep]
			else:
				new_det_pos = torch.zeros(0).cuda()
				new_det_features = torch.zeros(0).cuda()
		return new_det_pos, new_det_features

	def clear_inactive(self):
		to_remove = []
		for t in self.inactive_tracks:
			if t.is_to_purge():
				to_remove.append(t)
		for t in to_remove:
			self.inactive_tracks.remove(t)

	def get_appearances(self, blob):
		old_features = self.get_features()
		new_features = self.cnn.test_rois(blob['data'][0], self.get_pos()).data
		scores = 1 - F.cosine_similarity(old_features, new_features, dim=1)
		return scores.view(-1,1), new_features

	def set_features(self, new_features):
		for t,f in zip(self.tracks, new_features):
			t.features = f.view(1,-1)

	def step(self, blob):

		cl = 1

		###########################
		# Look for new detections #
		###########################
		_, scores, bbox_pred, rois = self.frcnn.test_image(blob['data'][0], blob['im_info'][0])
		#_, _, _, _ = self.frcnn.test_image(blob['data'][0], blob['im_info'][0])
		if self.use_detections:
			dets = blob['dets']
			if len(dets) > 0:
				dets = torch.cat(dets, 0)			
				_, scores, bbox_pred, rois = self.frcnn.test_rois(dets)
			else:
				rois = torch.zeros(0).cuda()

		if rois.nelement() > 0:
			boxes = bbox_transform_inv(rois, bbox_pred)
			boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data

			# Filter out tracks that have too low person score
			scores = scores[:,cl]
			inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)
		else:
			inds = torch.zeros(0).cuda()

		if inds.nelement() > 0:
			boxes = boxes[inds]
			det_pos = boxes[:,cl*4:(cl+1)*4]
			det_scores = scores[inds]
		else:
			det_pos = torch.zeros(0).cuda()
			det_scores = torch.zeros(0).cuda()

		##################
		# Predict tracks #
		##################
		num_tracks = 0
		nms_inp_reg = torch.zeros(0).cuda()
		if len(self.tracks) > 0:

			self.regress_tracks(blob)
			
			if len(self.tracks) > 0:
				
				# create nms input
				appearance_scores, new_features = self.get_appearances(blob)
				nms_inp_reg = torch.cat((self.get_pos(), appearance_scores.add(3)), 1)

				# nms here if tracks overlap
				#keep = nms(nms_inp_reg, self.regression_nms_thresh)
				keep = torch.lt(appearance_scores.view(-1), self.id_threshold).nonzero()
				if keep.nelement() > 0:
					keep = keep[:,0]
					self.keep(keep)
					nms_inp_reg = nms_inp_reg[keep]
					new_features[keep]
					self.set_features(new_features)
					num_tracks = nms_inp_reg.size(0)
				else:
					keep = []
					self.keep(keep)
					nms_inp_reg = torch.zeros(0).cuda()
					num_tracks = 0

			else:
				self.reset(hard=False)

		#####################
		# Create new tracks #
		#####################

		# create nms input and nms new detections
		if det_pos.nelement() > 0:
			nms_inp_det = torch.cat((det_pos, det_scores.view(-1,1)), 1)
		else:
			nms_inp_det = torch.zeros(0).cuda()
		if nms_inp_det.nelement() > 0:
			keep = nms(nms_inp_det, self.detection_nms_thresh)
			nms_inp_det = nms_inp_det[keep]
			# check with every track in a single run (problem if tracks delete each other)
			for i in range(num_tracks):
				nms_inp = torch.cat((nms_inp_reg[i].view(1,-1), nms_inp_det), 0)
				keep = nms(nms_inp, self.detection_nms_thresh)
				keep = keep[torch.ge(keep,1)]
				if keep.nelement() == 0:
					nms_inp_det = nms_inp_det.new(0)
					break
				nms_inp_det = nms_inp[keep]

		if nms_inp_det.nelement() > 0:
			new_det_pos = nms_inp_det[:,:4]

			# try to redientify tracks
			new_det_pos, new_det_features = self.reid(blob, new_det_pos)

			# add new
			if new_det_pos.nelement() > 0:
				self.add(new_det_pos, new_det_features)

		####################
		# Generate Results #
		####################

		for t in self.tracks:
			track_ind = int(t.id)
			if track_ind not in self.results.keys():
				self.results[track_ind] = {}
			pos = t.pos[0] / blob['im_info'][0][2]
			self.results[track_ind][self.im_index] = pos.cpu().numpy()

		self.im_index += 1

		self.clear_inactive()

		#print("tracks active: {}/{}".format(num_tracks, self.track_num))
		#print("len active: {}\nlen inactive: {}".format(len(self.tracks), len(self.inactive_tracks)))

	def get_results(self):
		return self.results


class Track(object):

	def __init__(self, pos, track_id, features, regression_person_thresh, inactive_patience):
		self.id = track_id
		self.pos = pos
		self.features = features
		self.count_inactive = 0
		self.regression_person_thresh = regression_person_thresh
		self.inactive_patience = inactive_patience

	def is_dead(self, score):
		if score < self.regression_person_thresh:
			return True
		else:
			return False

	def is_to_purge(self):
		self.count_inactive += 1
		if self.count_inactive > self.inactive_patience:
			return True
		else:
			return False
