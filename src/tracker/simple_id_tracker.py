from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms

import torch
from torch.autograd import Variable
import numpy as np

class Simple_ID_Tracker():

	def __init__(self, frcnn, detection_person_thresh, regression_person_thresh, detection_nms_thresh,
		regression_nms_thresh, alive_patience, reid_module):
		self.frcnn = frcnn
		self.detection_person_thresh = detection_person_thresh
		self.regression_person_thresh = regression_person_thresh
		self.detection_nms_thresh = detection_nms_thresh
		self.regression_nms_thresh = regression_nms_thresh
		self.alive_patience = alive_patience
		self.reid_module = reid_module

		self.reset()

	def reset(self, hard=True):
		self.ind2track = torch.zeros(0).cuda()
		self.pos = torch.zeros(0).cuda()
		self.kill_counter = torch.zeros(0).cuda()
		self.hidden = Variable(torch.zeros(0)).cuda()
		self.cell_state = Variable(torch.zeros(0)).cuda()
		self.lstm_out = Variable(torch.zeros(0)).cuda()

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def keep(self, keep):
		self.pos = self.pos[keep]
		self.ind2track = self.ind2track[keep]
		self.kill_counter = self.kill_counter[keep]
		self.hidden = self.hidden[:,keep,:]
		self.cell_state = self.cell_state[:,keep,:]
		self.lstm_out = self.lstm_out[keep]

	def add(self, num_new, new_det_pos):
		self.pos = torch.cat((self.pos, new_det_pos), 0)
		self.ind2track = torch.cat((self.ind2track, torch.arange(self.track_num, self.track_num+num_new).cuda()), 0)
		self.track_num += num_new
		self.kill_counter = torch.cat((self.kill_counter, torch.zeros(num_new).cuda()), 0)
		# create new hidden states
		hidden, cell_state = self.reid_module.init_hidden(num_new)
		if self.hidden.nelement() > 0:
			self.hidden = torch.cat((self.hidden, hidden), 1)
			self.cell_state = torch.cat((self.cell_state, cell_state), 1)
		else:
			self.hidden = hidden
			self.cell_state = cell_state

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
		nms_inp_reg = self.pos.new(0)
		if self.pos.nelement() > 0:
			# regress
			_, _, bbox_pred, rois = self.frcnn.test_rois(self.pos)
			boxes = bbox_transform_inv(rois, bbox_pred)
			boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data
			self.pos = boxes[:,cl*4:(cl+1)*4]

			# get scores of new regressed positions
			_, scores, _, _ = self.frcnn.test_rois(self.pos)
			scores = scores[:,cl]

			# check if still is a valid person
			dead = torch.lt(scores, self.regression_person_thresh)
			self.kill_counter[dead] += 1
			self.kill_counter[~dead] = 0
			keep = torch.lt(self.kill_counter, self.alive_patience).nonzero()
			if keep.nelement() > 0:
				keep = keep[:,0]
				self.keep(keep)
				scores = scores[keep]

				# create nms input
				#nms_inp_reg = torch.cat((self.pos, self.pos.new(self.pos.size(0),1).fill_(2)),1)
				#nms_inp_reg = torch.cat((self.pos, scores.add_(2)),1)
				appearance_scores = self.reid_module.test_rois(blob['data'][0], self.pos, self.lstm_out)
				nms_inp_reg = torch.cat((self.pos, appearance_scores.data.add(2)), 1)

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

			num_new = nms_inp_det.size(0)
			new_det_pos = nms_inp_det[:,:4]

			# add new
			self.add(num_new, new_det_pos)

		if self.pos.nelement() > 0:
			self.lstm_out, (self.hidden, self.cell_state) = self.reid_module.feed_rois(blob['data'][0], self.pos, (self.hidden, self.cell_state))
			#_,_ = self.reid_module.feed_rois(blob['data'][0], self.pos, h)
			#print("out")

		####################
		# Generate Results #
		####################

		for j,t in enumerate(self.pos / blob['im_info'][0][2]):
			track_ind = int(self.ind2track[j])
			if track_ind not in self.results.keys():
				self.results[track_ind] = {}
			self.results[track_ind][self.im_index] = t.cpu().numpy()

		self.im_index += 1

		#print("tracks active: {}/{}".format(num_tracks, self.track_num))
		print(self.hidden.size())

	def get_results(self):
		return self.results