from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms

import torch
from torch.autograd import Variable
import numpy as np
import cv2
import os

class Simple_Align_Tracker():

	def __init__(self, frcnn, detection_person_thresh, regression_person_thresh, detection_nms_thresh,
		regression_nms_thresh, alive_patience, use_detections):
		self.frcnn = frcnn
		self.detection_person_thresh = detection_person_thresh
		self.regression_person_thresh = regression_person_thresh
		self.detection_nms_thresh = detection_nms_thresh
		self.regression_nms_thresh = regression_nms_thresh
		self.alive_patience = alive_patience
		self.use_detections = use_detections

		self.reset()

	def reset(self, hard=True):
		self.ind2track = torch.zeros(0).cuda()
		self.pos = torch.zeros(0).cuda()
		#self.features = torch.zeros(0).cuda()
		self.kill_counter = torch.zeros(0).cuda()

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0
			self.debug = {}

	def keep(self, keep):
		self.pos = self.pos[keep]
		#self.features = self.features[keep]
		self.ind2track = self.ind2track[keep]
		self.kill_counter = self.kill_counter[keep]

	def align(self, blob):
		if self.im_index > 0:
			im1 = self.last_image.cpu().numpy()
			im2 = blob['data'][0][0].cpu().numpy()
			im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
			im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
			sz = im1.shape
			warp_mode = cv2.MOTION_EUCLIDEAN
			warp_matrix = np.eye(2, 3, dtype=np.float32)
			#number_of_iterations = 5000
			#number_of_iterations = 20
			number_of_iterations = 10
			termination_eps = 1e-10
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
			(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
			warp_matrix = torch.from_numpy(warp_matrix)
			pos = []
			for p in self.pos:
				p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
				p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)
				p1_n = torch.mm(warp_matrix, p1).view(1,2)
				p2_n = torch.mm(warp_matrix, p2).view(1,2)
				pos.append(torch.cat((p1_n, p2_n), 1))
			self.pos = torch.cat(pos, 0).cuda()

	def step(self, blob):
		debug = {'reg':{}, 'kill_score':{}, 'kill_nms':{}, 'new':{}}

		cl = 1

		###########################
		# Look for new detections #
		###########################
		#_, scores, bbox_pred, rois = self.frcnn.test_image(blob['data'][0], blob['im_info'][0])
		_, _, _, _ = self.frcnn.test_image(blob['data'][0], blob['im_info'][0])
		if self.use_detections:
			dets = blob['dets']
			if len(dets) > 0:
				dets = torch.cat(dets, 0)			
				_, scores, bbox_pred, rois = self.frcnn.test_rois(dets)
				#boxes = bbox_transform_inv(rois, bbox_pred)
				#boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data
				#_, scores, bbox_pred, rois = self.frcnn.test_rois(dets)
			else:
				rois = torch.zeros(0).cuda()
		#else:
		#	boxes = bbox_transform_inv(rois, bbox_pred)
		#	boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data

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
		nms_inp_reg = self.pos.new(0)
		if self.pos.nelement() > 0:
			# align
			self.align(blob)
			# regress
			_, _, bbox_pred, rois = self.frcnn.test_rois(self.pos)
			boxes = bbox_transform_inv(rois, bbox_pred)
			boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data
			self.pos = boxes[:,cl*4:(cl+1)*4]

			# add new regression pos to debug
			for j,t in enumerate(self.pos / blob['im_info'][0][2]):
				track_ind = int(self.ind2track[j])
				debug['reg'][track_ind] = t.cpu().numpy()

			# get scores of new regressed positions
			_, scores, _, _ = self.frcnn.test_rois(self.pos)
			scores = scores[:,cl]

			# check if still is a valid person
			dead = torch.le(scores, self.regression_person_thresh)
			self.kill_counter[dead] += 1
			self.kill_counter[~dead] = 0
			keep = torch.lt(self.kill_counter, self.alive_patience).nonzero()

			not_keep = list(np.arange(0,len(self.pos)))
			tracks = []
			if keep.nelement() > 0:
				for i in keep[:,0]:
					not_keep.remove(i)
			for i in not_keep:
				track_ind = int(self.ind2track[i])
				debug['kill_score'][track_ind] = ""

			if keep.nelement() > 0:
				keep = keep[:,0]
				self.keep(keep)
				scores = scores[keep]

				# create nms input
				#nms_inp_reg = torch.cat((self.pos, self.pos.new(self.pos.size(0),1).fill_(2)),1)
				nms_inp_reg = torch.cat((self.pos, scores.add_(2).view(-1,1)),1)
				#nms_inp_reg = torch.cat((self.pos, torch.rand(scores.size()).add_(2).view(-1,1).cuda()),1)

				# nms here if tracks overlap
				keep = nms(nms_inp_reg, self.regression_nms_thresh)

				not_keep = list(np.arange(0,len(self.pos)))
				tracks = []
				for i in keep:
					not_keep.remove(i)
				for i in not_keep:
					track_ind = int(self.ind2track[i])
					debug['kill_nms'][track_ind] = ""

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

			num_new = nms_inp_det.size(0)
			new_det_pos = nms_inp_det[:,:4]

			for i,p in enumerate(new_det_pos):
				debug['new'][i+self.track_num] = p.cpu().numpy()

			self.pos = torch.cat((self.pos, new_det_pos), 0)

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

		self.debug[self.im_index] = debug
		self.im_index += 1
		self.last_image = blob['data'][0][0]

		#print("tracks active: {}/{}".format(num_tracks, self.track_num))

	def get_results(self):
		return self.results

	def write_debug(self, path=None):
		lines = []
		for i,im in self.debug.items():
			lines.append("{}\n".format(i))

			for ii,case in im.items():
				lines.append("\t{}\n".format(ii))
				for track, val in case.items():
					lines.append("\t\t{}    {}\n".format(track, val))
		#print(lines)
		with open(path, "w") as file:
			for l in lines:
				file.write(l)