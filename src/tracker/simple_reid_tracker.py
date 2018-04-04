from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms

import torch
from torch.autograd import Variable
import numpy as np
from scipy.optimize import linear_sum_assignment

class Simple_ReID_Tracker():

	def __init__(self, frcnn, detection_person_thresh, regression_person_thresh, detection_nms_thresh,
		regression_nms_thresh, alive_patience, reid_module):
		self.frcnn = frcnn
		self.detection_person_thresh = detection_person_thresh
		self.regression_person_thresh = regression_person_thresh
		self.detection_nms_thresh = detection_nms_thresh
		self.regression_nms_thresh = regression_nms_thresh
		self.alive_patience = alive_patience
		self.reid_module = reid_module
		self.reid_threshold = 0.5
		self.inactive_patience = 10

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

	def add(self, new_det_pos):
		num_new = new_det_pos.size(0)
		for i in range(num_new):
			self.tracks.append(Track(self.reid_module, new_det_pos[i], self.track_num + i, self.regression_person_thresh, self.inactive_patience))
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
			t.pos = p

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
			pos = self.tracks[0].pos.view(1,-1)
		elif len(self.tracks) > 1:
			pos = torch.stack([t.pos for t in self.tracks],0)
		else:
			pos = torch.zeros(0).cuda()
		return pos

	def reid(self, image, new_det_pos):
		if len(self.inactive_tracks) >= 1:
			if len(self.inactive_tracks) == 1:
				costs = 1 - self.inactive_tracks[0].test_appearances(image, new_det_pos).view(1,-1)
			else:
				costs = 1 - torch.cat([t.test_appearances(image, new_det_pos).view(1,-1) for t in self.inactive_tracks], 0)
			row_ind, col_ind = linear_sum_assignment(costs.cpu().numpy())

			assigned = []
			remove_inactive = []
			for r,c in zip(row_ind, col_ind):
				if costs[r,c] <= 1 - self.reid_threshold:
					print("im: {}\ncosts: {}".format(self.im_index, costs[r,c]))
					t = self.inactive_tracks[r]
					self.tracks.append(t)
					t.count_dead = 0
					t.pos = new_det_pos[c]
					assigned.append(c)
					remove_inactive.append(t)

			for t in remove_inactive:
				self.inactive_tracks.remove(t)

			keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
			if keep.nelement() > 0:
				new_det_pos = new_det_pos[keep]
			else:
				new_det_pos = torch.zeros(0).cuda()
		return new_det_pos

	def clear_inactive(self):
		to_remove = []
		for t in self.inactive_tracks:
			if t.is_to_purge():
				to_remove.append(t)
		for t in to_remove:
			self.inactive_tracks.remove(t)

	def step(self, blob):

		cl = 1

		###########################
		# Look for new detections #
		###########################
		_, scores, bbox_pred, rois = self.frcnn.test_image(blob['data'][0], blob['im_info'][0])

		boxes = bbox_transform_inv(rois, bbox_pred)
		boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data

		# Filter out tracks that have too low person score
		scores = scores[:,cl]
		inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)
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
				appearance_scores = torch.cat([t.test_own_appearance(blob['data'][0]) for t in self.tracks])
				nms_inp_reg = torch.cat((self.get_pos(), appearance_scores.add(2)), 1)

				# nms here if tracks overlap
				keep = nms(nms_inp_reg, self.regression_nms_thresh)
				self.keep(keep)
				nms_inp_reg = nms_inp_reg[keep]

				# number of active tracks
				num_tracks = nms_inp_reg.size(0)
			else:
				self.reset(hard=False)

		#####################
		# Create new tracks #
		#####################

		# create nms input and nms new detections
		nms_inp_det = torch.cat((det_pos, det_scores.view(-1,1)), 1)
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
			new_det_pos = self.reid(blob['data'][0], new_det_pos)

			# add new
			if new_det_pos.nelement() > 0:
				self.add(new_det_pos)

		if len(self.tracks) > 0:
			for t in self.tracks:
				t.feed_appearance(blob['data'][0])

		####################
		# Generate Results #
		####################

		for t in self.tracks:
			track_ind = int(t.id)
			if track_ind not in self.results.keys():
				self.results[track_ind] = {}
			pos = t.pos / blob['im_info'][0][2]
			self.results[track_ind][self.im_index] = pos.cpu().numpy()

		self.im_index += 1

		self.clear_inactive()

		#print("tracks active: {}/{}".format(num_tracks, self.track_num))
		print("len active: {}\nlen inactive: {}".format(len(self.tracks), len(self.inactive_tracks)))

	def get_results(self):
		return self.results


class Track(object):

	def __init__(self, reid_module, pos, track_id, regression_person_thresh, inactive_patience):
		self.id = track_id
		self.pos = pos
		self.reid_module = reid_module
		self.hidden = self.reid_module.init_hidden(1)
		self.count_inactive = 0
		self.regression_person_thresh = regression_person_thresh
		self.inactive_patience = inactive_patience

	def feed_appearance(self, data):
		self.lstm_out, self.hidden = self.reid_module.feed_rois(data, self.pos.view(1,-1), self.hidden)

	def test_appearances(self, data, pos):
		return self.reid_module.test_rois(data, pos, self.lstm_out).data

	def test_own_appearance(self, data):
		return self.reid_module.test_rois(data, self.pos.view(1,-1), self.lstm_out).data

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