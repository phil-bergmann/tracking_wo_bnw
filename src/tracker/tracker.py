from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms
from .utils import bbox_overlaps

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor, Normalize

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from collections import deque
import cv2
import matplotlib.pyplot as plt
import os

class Tracker():
	"""The main tracking file, here is where magic happens."""

	def __init__(self, frcnn, cnn, detection_person_thresh, regression_person_thresh, detection_nms_thresh,
		regression_nms_thresh, public_detections, do_reid, inactive_patience, do_align, reid_sim_threshold,
		max_features_num, reid_iou_threshold, warp_mode, number_of_iterations, termination_eps, motion_model):
		self.frcnn = frcnn
		self.cnn = cnn
		self.detection_person_thresh = detection_person_thresh
		self.regression_person_thresh = regression_person_thresh
		self.detection_nms_thresh = detection_nms_thresh
		self.regression_nms_thresh = regression_nms_thresh
		self.public_detections = public_detections
		self.inactive_patience = inactive_patience
		self.do_reid = do_reid
		self.max_features_num = max_features_num
		self.reid_sim_threshold = reid_sim_threshold
		self.reid_iou_threshold = reid_iou_threshold
		self.do_align = do_align
		self.motion_model = motion_model

		self.warp_mode = eval(warp_mode)
		self.number_of_iterations = number_of_iterations
		self.termination_eps = termination_eps

		self.reset()

	def reset(self, hard=True):
		self.tracks = []
		self.inactive_tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def keep(self, keep):
		"""Keeps only the tracks defined by keep alive and moves the remaining to inactive."""
		tracks = []
		for i in keep:
			tracks.append(self.tracks[i])
		new_inactive = [t for t in self.tracks if t not in tracks]
		self.inactive_tracks += new_inactive
		self.tracks = tracks

	def add(self, new_det_pos, new_det_scores, new_det_features):
		"""Initializes new Track objects and saves them."""
		num_new = new_det_pos.size(0)
		for i in range(num_new):
			self.tracks.append(Track(new_det_pos[i].view(1,-1), new_det_scores[i], self.track_num + i, new_det_features[i].view(1,-1),
																	self.inactive_patience, self.max_features_num))
		self.track_num += num_new

	def regress_tracks(self, blob):
		"""Regress the position of the tracks and also checks their scores."""
		cl = 1

		pos = self.get_pos()

		# regress
		_, scores, bbox_pred, rois = self.frcnn.test_rois(pos)
		boxes = bbox_transform_inv(rois, bbox_pred)
		boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data
		pos = boxes[:,cl*4:(cl+1)*4]
		scores = scores[:,cl]

		s = []
		for i in range(len(self.tracks)-1,-1,-1):
			t = self.tracks[i]
			t.score = scores[i]
			if scores[i] <= self.regression_person_thresh:
				self.tracks.remove(t)
				self.inactive_tracks.append(t)
			else:
				s.append(scores[i])
				t.pos = pos[i].view(1,-1)
		return torch.Tensor(s[::-1]).cuda()

	def get_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.tracks) == 1:
			pos = self.tracks[0].pos
		elif len(self.tracks) > 1:
			pos = torch.cat([t.pos for t in self.tracks],0)
		else:
			pos = torch.zeros(0).cuda()
		return pos

	def get_features(self):
		"""Get the features of all active tracks."""
		if len(self.tracks) == 1:
			features = self.tracks[0].features
		elif len(self.tracks) > 1:
			features = torch.cat([t.features for t in self.tracks],0)
		else:
			features = torch.zeros(0).cuda()
		return features
	
	def get_inactive_features(self):
		"""Get the features of all inactive tracks."""
		if len(self.inactive_tracks) == 1:
			features = self.inactive_tracks[0].features
		elif len(self.inactive_tracks) > 1:
			features = torch.cat([t.features for t in self.inactive_tracks],0)
		else:
			features = torch.zeros(0).cuda()
		return features

	def reid(self, blob, new_det_pos, new_det_scores):
		"""Tries to ReID inactive tracks with provided detections."""
		new_det_features = self.cnn.test_rois(blob['app_data'][0], new_det_pos / blob['im_info'][0][2]).data
		if len(self.inactive_tracks) >= 1 and self.do_reid:
			# calculate appearance distances
			dist_mat = []
			pos = []
			for t in self.inactive_tracks:
				dist_mat.append(torch.cat([t.test_features(feat.view(1,-1)) for feat in new_det_features], 1))
				pos.append(t.pos)
			if len(dist_mat) > 1:
				dist_mat = torch.cat(dist_mat, 0)
				pos = torch.cat(pos,0)
			else:
				dist_mat = dist_mat[0]
				pos = pos[0]

			# calculate IoU distances
			iou = bbox_overlaps(pos, new_det_pos)
			iou_mask = torch.ge(iou, self.reid_iou_threshold)
			iou_neg_mask = ~iou_mask
			# make all impossible assignemnts to the same add big value
			dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float()*1000
			dist_mat = dist_mat.cpu().numpy()

			row_ind, col_ind = linear_sum_assignment(dist_mat)

			assigned = []
			remove_inactive = []
			for r,c in zip(row_ind, col_ind):
				if dist_mat[r,c] <= self.reid_sim_threshold:
					#print("im: {}\ncosts: {}".format(self.im_index, costs[r,c]))
					t = self.inactive_tracks[r]
					self.tracks.append(t)
					t.count_inactive = 0
					t.last_v = torch.Tensor([])
					t.pos = new_det_pos[c].view(1,-1)
					t.add_features(new_det_features[c].view(1,-1))
					assigned.append(c)
					remove_inactive.append(t)

			for t in remove_inactive:
				self.inactive_tracks.remove(t)
				#print("matched in frame {}".format(self.im_index))

			keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
			if keep.nelement() > 0:
				new_det_pos = new_det_pos[keep]
				new_det_scores = new_det_scores[keep]
				new_det_features = new_det_features[keep]
			else:
				new_det_pos = torch.zeros(0).cuda()
				new_det_scores = torch.zeros(0).cuda()
				new_det_features = torch.zeros(0).cuda()
		return new_det_pos, new_det_scores, new_det_features

	def clear_inactive(self):
		"""Checks if inactive tracks should be removed."""
		to_remove = []
		for t in self.inactive_tracks:
			if t.is_to_purge():
				to_remove.append(t)
		for t in to_remove:
			self.inactive_tracks.remove(t)

	def get_appearances(self, blob):
		"""Uses the siamese CNN to get the features for all active tracks."""
		new_features = self.cnn.test_rois(blob['app_data'][0], self.get_pos() / blob['im_info'][0][2]).data
		return new_features

	def add_features(self, new_features):
		"""Adds new appearance features to active tracks."""
		for t,f in zip(self.tracks, new_features):
			t.add_features(f.view(1,-1))

	def align(self, blob):
		"""Aligns the positions of active and inactive tracks depending on camera motion."""
		if self.im_index > 0:
			im1 = self.last_image.cpu().numpy()
			im2 = blob['data'][0][0].cpu().numpy()
			im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
			im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
			sz = im1.shape
			warp_mode = self.warp_mode
			warp_matrix = np.eye(2, 3, dtype=np.float32)
			#number_of_iterations = 5000
			number_of_iterations = self.number_of_iterations
			termination_eps = self.termination_eps
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
			(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
			warp_matrix = torch.from_numpy(warp_matrix)
			pos = []
			for t in self.tracks:
				p = t.pos[0]
				p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
				p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)

				p1_n = torch.mm(warp_matrix, p1).view(1,2)
				p2_n = torch.mm(warp_matrix, p2).view(1,2)
				pos = torch.cat((p1_n, p2_n), 1).cuda()

				t.pos = pos.view(1,-1)
				#t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data

			if self.do_reid:
				for t in self.inactive_tracks:
					p = t.pos[0]
					p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
					p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)
					p1_n = torch.mm(warp_matrix, p1).view(1,2)
					p2_n = torch.mm(warp_matrix, p2).view(1,2)
					pos = torch.cat((p1_n, p2_n), 1).cuda()
					t.pos = pos.view(1,-1)

			if self.motion_model:
				for t in self.tracks:
					if t.last_pos.nelement() > 0:
						p = t.last_pos[0]
						p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
						p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)

						p1_n = torch.mm(warp_matrix, p1).view(1,2)
						p2_n = torch.mm(warp_matrix, p2).view(1,2)
						pos = torch.cat((p1_n, p2_n), 1).cuda()

						t.last_pos = pos.view(1,-1)

	def motion(self):
		"""Applies a simple linear motion model that only consideres the positions at t-1 and t-2."""
		for t in self.tracks:
			last_pos = t.pos.clone()
			if t.last_pos.nelement() > 0:
				# extract center coordinates of last pos
				x1l = t.last_pos[0,0]
				y1l = t.last_pos[0,1]
				x2l = t.last_pos[0,2]
				y2l = t.last_pos[0,3]
				cxl = (x2l + x1l)/2
				cyl = (y2l + y1l)/2

				# extract coordinates of current pos
				x1p = t.pos[0,0]
				y1p = t.pos[0,1]
				x2p = t.pos[0,2]
				y2p = t.pos[0,3]
				cxp = (x2p + x1p)/2
				cyp = (y2p + y1p)/2
				wp = x2p - x1p
				hp = y2p - y1p

				# v = cp - cl, x_new = v + cp = 2cp - cl
				cxp_new = 2*cxp - cxl
				cyp_new = 2*cyp - cyl

				t.pos[0,0] = cxp_new - wp/2
				t.pos[0,1] = cyp_new - hp/2
				t.pos[0,2] = cxp_new + wp/2
				t.pos[0,3] = cyp_new + hp/2

				t.last_v = torch.Tensor([cxp - cxl, cyp - cyl]).cuda()
			t.last_pos = last_pos
		
		if self.do_reid:
			for t in self.inactive_tracks:
				if t.last_v.nelement() > 0:
					# extract coordinates of current pos
					x1p = t.pos[0,0]
					y1p = t.pos[0,1]
					x2p = t.pos[0,2]
					y2p = t.pos[0,3]
					cxp = (x2p + x1p)/2
					cyp = (y2p + y1p)/2
					wp = x2p - x1p
					hp = y2p - y1p

					cxp_new = cxp + t.last_v[0]
					cyp_new = cyp + t.last_v[1]

					t.pos[0,0] = cxp_new - wp/2
					t.pos[0,1] = cyp_new - hp/2
					t.pos[0,2] = cxp_new + wp/2
					t.pos[0,3] = cyp_new + hp/2

	def step(self, blob):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""

		# only the class person used here
		cl = 1

		###########################
		# Look for new detections #
		###########################
		self.frcnn.load_image(blob['data'][0], blob['im_info'][0])
		if self.public_detections:
			dets = blob['dets']
			if len(dets) > 0:
				dets = torch.cat(dets, 0)[:,:4]		
				_, scores, bbox_pred, rois = self.frcnn.test_rois(dets)
			else:
				rois = torch.zeros(0).cuda()
		else:
			_, scores, bbox_pred, rois = self.frcnn.detect()

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
			# align
			if self.do_align:
				self.align(blob)
			# apply motion model
			if self.motion_model:
				self.motion()
			#regress
			person_scores = self.regress_tracks(blob)
			
			if len(self.tracks) > 0:
				
				# create nms input
				new_features = self.get_appearances(blob)

				# nms here if tracks overlap
				nms_inp_reg = torch.cat((self.get_pos(), person_scores.add_(3).view(-1,1)),1)
				keep = nms(nms_inp_reg, self.regression_nms_thresh)

				# Plot the killed tracks for debugging
				not_keep = list(np.arange(0,len(self.tracks)))
				tracks = []
				for i in keep:
					not_keep.remove(i)
				
				if keep.nelement() > 0:
					self.keep(keep)
					nms_inp_reg = nms_inp_reg[keep]
					new_features = new_features[keep]
					self.add_features(new_features)
					num_tracks = nms_inp_reg.size(0)
				else:
					keep = []
					self.keep(keep)
					nms_inp_reg = torch.zeros(0).cuda()
					num_tracks = 0

			else:
				pass
				#self.reset(hard=False)

		#####################
		# Create new tracks #
		#####################

		# !!! Here NMS is used to filter out detections that are already covered by tracks. This is
		# !!! done by iterating through the active tracks one by one, assigning them a bigger score
		# !!! than 1 (maximum score for detections) and then filtering the detections with NMS.
		# !!! In the paper this is done by calculating the overlap with existing tracks, but the
		# !!! result stays the same.
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
			new_det_scores = nms_inp_det[:,4]

			# try to redientify tracks
			new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

			# add new
			if new_det_pos.nelement() > 0:
				self.add(new_det_pos, new_det_scores, new_det_features)

		####################
		# Generate Results #
		####################

		for t in self.tracks:
			track_ind = int(t.id)
			if track_ind not in self.results.keys():
				self.results[track_ind] = {}
			pos = t.pos[0] / blob['im_info'][0][2]
			sc = t.score
			self.results[track_ind][self.im_index] = np.concatenate([pos.cpu().numpy(), np.array([sc])])

		self.im_index += 1
		self.last_image = blob['data'][0][0]

		self.clear_inactive()

		#print("tracks active: {}/{}".format(num_tracks, self.track_num))
		#print("len active: {}\nlen inactive: {}".format(len(self.tracks), len(self.inactive_tracks)))

	def get_results(self):
		return self.results


class Track(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num):
		self.id = track_id
		self.pos = pos
		self.score = score
		self.features = deque([features])
		self.ims = deque([])
		self.count_inactive = 0
		self.inactive_patience = inactive_patience
		self.max_features_num = max_features_num
		self.last_pos = torch.Tensor([])
		self.last_v = torch.Tensor([])

	def is_to_purge(self):
		"""Tests if the object has been too long inactive and is to remove."""
		self.count_inactive += 1
		self.last_pos = torch.Tensor([])
		if self.count_inactive > self.inactive_patience:
			return True
		else:
			return False

	def add_features(self, features):
		"""Adds new appearance features to the object."""
		self.features.append(features)
		if len(self.features) > self.max_features_num:
			self.features.popleft()

	def test_features(self, test_features):
		"""Compares test_features to features of this Track object"""
		if len(self.features) > 1:
			features = torch.cat(self.features, 0)
		else:
			features = self.features[0]
		features = features.mean(0, keepdim=True)
		dist = F.pairwise_distance(features, test_features)
		return dist
