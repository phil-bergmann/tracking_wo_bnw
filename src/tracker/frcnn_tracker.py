
from model.config import cfg as frcnn_cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms

import torch
from torch.autograd import Variable
import numpy as np

from utils import boxes2rois

class FRCNN_TRACKER():

	def __init__(self, frcnn):
		self.frcnn = frcnn
		self.track_num = 0
		self.pos2track = np.arange(0)
		self.cls_dets = np.zeros((0,5))
		self.results = {}
		self.im_index = 0
		self.fc7 = np.zeros((0,4096))

	def step(self, blobs):
	
		#blobs = data_layer.forward()

		#im_shape = blobs['im_info'][0,:2]/blobs['im_info'][0,2]

		cl = 1

		##################
		# Predict tracks #
		##################

		if self.cls_dets.shape[0] > 0:
			rois_score = Variable(torch.zeros(self.cls_dets.shape[0],1)).cuda()
			rois_bb = Variable(torch.from_numpy(self.cls_dets[:,0:4])).cuda()
			rois_old = torch.cat((rois_score, rois_bb),1)
			rois = boxes2rois(self.cls_dets[:,0:4])

			_, _, bbox_pred, rois = self.frcnn.test_image(blobs['data'][0], blobs['im_info'], rois)

			boxes = bbox_transform_inv(rois[:,1:5], bbox_pred)
			boxes = clip_boxes(boxes, blobs['im_info'][0,:2])

			# get scores of new regressed positions
			rois_pred = boxes2rois(boxes)
			_, scores, _, rois = self.frcnn.test_image(blobs['data'][0], blobs['im_info'], rois_pred)
			fc7 = self.frcnn.get_fc7()


			# check if still is a valid person
			inds = torch.gt(scores, 0.01).nonzero().view(-1)
			self.pos2track = self.pos2track[inds]
			self.fc7 = self.fc7[inds]
			fc7 = fc7[inds]
			boxes = boxes[inds]
			cls_boxes = boxes[:,cl*4:(cl+1)*4]
			#self.cls_dets = np.hstack((cls_boxes, np.arange(cls_boxes.shape[0]+1,1,-1)[:, np.newaxis])) \
			#	.astype(np.float32, copy=False)
			#self.cls_dets = np.hstack((cls_boxes, scores[inds, cl][:, np.newaxis]+2)) \
			#	.astype(np.float32, copy=False)

			# compare fc7 features
			fc7_score = torch.sum(torch.abs(fc7-self.fc7),1, True) + 2
			self.cls_dets = torch.cat((cls_boxes, fc7_score),1)
			

			# now do nms to check if tracks now fall together
			#keep = nms(torch.from_numpy(self.cls_dets), frcnn_cfg.TEST.NMS).numpy() if self.cls_dets.size > 0 else []
			if self.cls_dets.size()[0] > 0:
				keep = nms(self.cls_dets, 0.7)
				keep = torch.sort(keep)

				self.cls_dets = self.cls_dets[keep]
				self.pos2track = self.pos2track[keep]
				self.fc7 = self.fc7[keep]

		# number of old tracks
		old_tracks = self.cls_dets.size()[0]

		###########################
		# Look for new detections #
		###########################
		_, scores, bbox_pred, rois = self.frcnn.test_image(blobs['data'][0], blobs['im_info'])
		fc7 = self.frcnn.get_fc7()
		
		# Apply bounding-box regression deltas
		boxes = bbox_transform_inv(rois[:,1:5], bbox_pred)
		boxes = clip_boxes(boxes, blobs['im_info'][0,:2])

		#inds = np.where(scores[:, cl] > 0.05)[0]
		#cls_scores = scores[inds, cl]
		#cls_boxes = boxes[inds, cl*4:(cl+1)*4]
		#fc7 = fc7[inds]
		inds = torch.gt(scores, 0.01).nonzero().view(-1)
		scores = scores[inds]
		cls_scores = scores[:,cl]
		boxes = boxes[inds]
		cls_boxes = boxes[:,cl*4:(cl+1)*4]

		cls_dets_new = torch.cat((cls_boxes, cls_scores.view(-1,1)),1)

		# nms to see if new detections are already taken care by prediction step
		self.cls_dets = torch.cat((self.cls_dets, cls_dets_new),0)

		if self.cls_dets.size()[0] > 0:

			keep = nms(self.cls_dets, frcnn_cfg.TEST.NMS)
			keep = torch.sort(keep)

			# only filter out new detections
			keep = torch.cat((torch.arange(old_tracks,out=torch.LongTensor()), keep[torch.ge(keep,old_tracks).nonzero().view(-1)]))

			fc7 = fc7[keep[torch.ge(keep,old_tracks).nonzero().view(-1)]-old_tracks]
			#print(fc7.shape)

			#print(keep)
			self.cls_dets = self.cls_dets[keep]

			# construct new mapping pos 2 track for new detections
			new_tracks = self.cls_dets.size()[0] - old_tracks
			self.pos2track = torch.cat((self.pos2track, torch.arange(self.track_num, self.track_num+new_tracks, out=torch.LongTensor())))
			self.track_num += new_tracks


		for j,t in enumerate(self.cls_dets[:,0:4] / blobs['im_info'][0,2]):
			track_ind = self.pos2track[j]
			if track_ind not in self.results.keys():
				self.results[track_ind] = {}
				#self.results[track_ind]['fc7'] = fc7[j-old_tracks]
				self.fc7 = np.cat((self.fc7, fc7[j-old_tracks]),0)
			self.results[track_ind][self.im_index] = t

		self.im_index += 1

		#print("{}\n{}".format(self.im_index-1,self.cls_dets))

	def get_results(self):
		return self.results
