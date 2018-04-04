import torch
from torch.autograd import Variable
import numpy as np

from model.bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms

from .lstm_regressor import increase_search_region

class Tracker():

	def __init__(self, frcnn, regressor, detection_thresh=0.3, nms_thresh=0.3, alive_thresh=0.5, alive_patience=1):
		self.frcnn = frcnn
		self.regressor = regressor

		self.search_region_factor = self.regressor.search_region_factor

		self.detection_thresh = detection_thresh
		self.nms_thresh = nms_thresh
		self.alive_thresh = alive_thresh
		self.alive_patience = alive_patience

		self.reset()

	def reset(self, hard=True):
		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

		self.ind2track = torch.zeros(0).cuda()
		self.features = torch.zeros(0).cuda()
		self.pos = torch.zeros(0).cuda()
		self.hidden = Variable(torch.zeros(0)).cuda()
		self.cell_state = Variable(torch.zeros(0)).cuda()
		self.scores = torch.zeros(0).cuda()
		self.kill_counter = torch.zeros(0).cuda()

	def keep(self, keep):
		self.pos = self.pos[keep]
		self.hidden = self.hidden[:,keep,:]
		self.cell_state = self.cell_state[:,keep,:]
		self.features = self.features[keep]
		self.ind2track = self.ind2track[keep]
		self.scores = self.scores[keep]
		self.kill_counter = self.kill_counter[keep]

	def step(self, blob, cnn=None):
		cl = 1

		###########################
		# Look for new detections #
		###########################
		_, scores, bbox_pred, rois = self.frcnn.test_image(blob['data'][0], blob['im_info'][0])

		boxes = bbox_transform_inv(rois, bbox_pred)
		boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data

		# Filter out tracks that have too low person score
		scores = scores[:,cl]
		inds = torch.gt(scores, self.detection_thresh)
		boxes = boxes[inds.nonzero().view(-1)]
		det_pos = boxes[:,cl*4:(cl+1)*4]
		det_scores = scores[inds]

		##################
		# Regress Tracks #
		##################
		num_tracks = 0
		nms_inp_reg = self.pos.new(0)
		if self.pos.nelement() > 0:
			# First extract features in new image on old pos
			if self.search_region_factor != 1.0:
				self.pos = increase_search_region(self.pos, self.search_region_factor)
				self.pos = clip_boxes(Variable(self.pos), blob['im_info'][0][:2]).data

			if cnn:
				search_features = cnn.test_rois(blob['data'][0], self.pos).data
			else:
				_, _, _, _ = self.frcnn.test_rois(self.pos)
				search_features = self.frcnn.get_fc7()

			# generate input and call Regressor
			print("here")
			print(self.features.size())
			print(self.scores.size())
			bbox_reg, alive, (self.hidden, self.cell_state) = self.regressor(Variable(self.features), Variable(self.scores),
				(self.hidden, self.cell_state), Variable(search_features))

			# kill tracks that are marked as dead
			dead = torch.lt(alive, self.alive_thresh).data
			self.kill_counter[dead] += 1
			self.kill_counter[~dead] = 0
			keep = torch.lt(self.kill_counter,self.alive_patience).nonzero()
			if keep.nelement() > 0:
				keep = keep[:,0]
				# filter variables
				self.keep(keep)
				bbox_reg = bbox_reg[keep]

				# now regress with the output of the Regressor
				boxes = bbox_transform_inv(self.pos, bbox_reg.data)
				boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data
				self.pos = boxes

				# get the features and scores at the regressed positions
				_, scores, _, _ = self.frcnn.test_rois(self.pos)
				if cnn:
					self.features = cnn.test_rois(blob['data'][0], self.pos).data
				else:
					self.features = self.frcnn.get_fc7()
				self.scores = scores[:,cl].contiguous().view(-1,1)

				# create nms input
				nms_inp_reg = torch.cat((self.pos, self.pos.new(self.pos.size(0),1).fill_(2)),1)

				# number of active tracks
				num_tracks = nms_inp_reg.size(0)
			else:
				self.reset(hard=False)


		#####################
		# Create new tracks #
		#####################

		# create nms input and nms new detections
		nms_inp_det = torch.cat((det_pos, det_scores.view(-1,1)), 1)
		keep = nms(nms_inp_det, self.nms_thresh)
		nms_inp_det = nms_inp_det[keep]
		#nms_inp = torch.cat((nms_inp_reg, nms_inp_det), 0)
		# check with every track in a single run (problem if tracks delete each other)
		for i in range(num_tracks):
			nms_inp = torch.cat((nms_inp_reg[i].view(1,-1), nms_inp_det), 0)
			keep = nms(nms_inp, self.nms_thresh)
			keep = keep[torch.ge(keep,1)]
			if keep.nelement() == 0:
				nms_inp_det = nms_inp_det.new(0)
				break
			nms_inp_det = nms_inp[keep]

		if nms_inp_det.nelement() > 0:

			num_new = nms_inp_det.size(0)
			new_det_pos = nms_inp_det[:,:4]

			self.pos = torch.cat((self.pos, new_det_pos), 0)
			# get the regressed features
			_, scores, _, _ = self.frcnn.test_rois(new_det_pos)
			if cnn:
				det_features = cnn.test_rois(blob['data'][0], new_det_pos).data
			else:
				det_features = self.frcnn.get_fc7()
			self.features = torch.cat((self.features, det_features), 0)
			self.scores = torch.cat((self.scores, scores[:,cl].contiguous().view(-1,1)), 0)

			# create new hidden states
			hidden, cell_state = self.regressor.init_hidden(num_new)
			if self.hidden.nelement() > 0:
				self.hidden = torch.cat((self.hidden, hidden), 1)
				self.cell_state = torch.cat((self.cell_state, cell_state), 1)
			else:
				self.hidden = hidden
				self.cell_state = cell_state

			self.ind2track = torch.cat((self.ind2track, torch.arange(self.track_num, self.track_num+num_new).cuda()), 0)

			self.track_num += num_new

			self.kill_counter = torch.cat((self.kill_counter, torch.zeros(num_new).cuda()), 0)


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

	def get_results(self):
		return self.results