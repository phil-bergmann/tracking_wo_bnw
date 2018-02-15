import torch
from torch.autograd import Variable
import numpy as np

from model.bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms

class Tracker():

	def __init__(self, frcnn, regressor, detection_thresh=0.3, nms_thresh=0.3):
		self.frcnn = frcnn
		self.regressor = regressor

		self.detection_thresh = detection_thresh
		self.nms_thresh = nms_thresh

	def reset(self):
		self.track_num = 0
		self.ind2track = torch.zeros(0).cuda()
		self.results = {}
		self.im_index = 0

		self.features = torch.zeros(0).cuda()
		self.pos = torch.zeros(0).cuda()
		self.hidden = Variable(torch.zeros(0)).cuda()
		self.cell_state = Variable(torch.zeros(0)).cuda()

	def step(self, blob):
		cl = 1

		###########################
		# Look for new detections #
		###########################
		_, scores, bbox_pred, rois = self.frcnn.test_image(blob['data'][0], blob['im_info'][0])
		fc7 = self.frcnn.get_fc7()

		boxes = bbox_transform_inv(rois, bbox_pred)
		boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data

		# Filter out tracks that have too low person score
		scores = scores[:,cl]
		inds = torch.gt(scores, self.detection_thresh)
		boxes = boxes[inds.nonzero().view(-1)]
		det_pos = boxes[:,cl*4:(cl+1)*4]
		# probably better generate the regressed features later
		#det_features = fc7[inds]
		det_scores = scores[inds]

		##################
		# Regress Tracks #
		##################
		num_tracks = 0
		nms_inp_reg = self.pos.new(0)
		if self.pos.nelement() > 0:
			# First extract fc7 features in new image on old pos, do here and not one-by-one in
			# LSTM for parallelism
			_, _, _, _ = self.frcnn.test_rois(self.pos)
			search_features = self.frcnn.get_fc7()

			# generate input and call Regressor
			input = Variable(torch.cat((self.features, search_features),1).view(1,-1,4096*2))
			bbox_reg, alive, (self.hidden, self.cell_state) = self.regressor(input, (self.hidden, self.cell_state))

			# now regress with the output of the Regressor
			boxes = bbox_transform_inv(self.pos, bbox_reg.data)
			boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data
			self.pos = boxes

			# get the features at the regressed positions
			_, _, _, _ = self.frcnn.test_rois(self.pos)
			self.features = self.frcnn.get_fc7()

			# create nms input
			nms_inp_reg = torch.cat((self.pos, self.pos.new(self.pos.size(0),1).fill_(2)),1)

			# number of active tracks
			num_tracks = nms_inp_reg.size(0)

		#####################
		# Create new tracks #
		#####################

		# create nms input
		nms_inp_det = torch.cat((det_pos, det_scores.view(-1,1)), 1)
		nms_inp = torch.cat((nms_inp_reg, nms_inp_det), 0)

		# apply nms
		keep = nms(nms_inp, self.nms_thresh)
		keep, _ = torch.sort(keep)

		# only filter out new detections
		#keep = torch.cat((torch.arange(0,num_tracks,out=torch.cuda.LongTensor()), keep[torch.ge(keep,old_tracks)]))

		# only new detections interesting for nms
		keep = keep[torch.ge(keep,num_tracks)] - num_tracks

		if keep.nelement() > 0:

			num_new = keep.nelement()

			self.pos = torch.cat((self.pos, det_pos[keep]), 0)
			# get the regressed features
			_, _, _, _ = self.frcnn.test_rois(det_pos[keep])
			det_features = self.frcnn.get_fc7()
			self.features = torch.cat((self.features, det_features), 0)

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


		####################
		# Generate Results #
		####################

		for j,t in enumerate(self.pos / blob['im_info'][0][2]):
			track_ind = int(self.ind2track[j])
			if track_ind not in self.results.keys():
				self.results[track_ind] = {}
			self.results[track_ind][self.im_index] = t.cpu().numpy()

		self.im_index += 1

	def get_results(self):
		return self.results