
from model.config import cfg as frcnn_cfg
from model.bbox_transform import bbox_transform_inv
from model.test import _clip_boxes
from model.nms_wrapper import nms

import torch
from torch.autograd import Variable
import numpy as np

class FRCNN_TRACKER():

	def __init__(self, frcnn, regressor):
		self.frcnn = frcnn
		self.regressor = regressor
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
			# extend rois to account for big displacements
			#fac = 10
			#self.cls_dets[:,0] -= fac
			#self.cls_dets[:,1] -= fac
			#self.cls_dets[:,2] += fac
			#self.cls_dets[:,3] += fac
			rois_bb = Variable(torch.from_numpy(self.cls_dets[:,0:4])).cuda()
			rois_old = torch.cat((rois_score, rois_bb),1)

			_, _, bbox_pred, rois = self.frcnn.test_image(blobs['data'][0], blobs['im_info'], rois_old)

			##### use new regressor

			bbox_pred = self.regressor.forward(Variable(torch.from_numpy(self.frcnn.get_fc7())).cuda()).data.cpu().numpy()
			#print(bbox_pred)

			#####
			boxes = bbox_transform_inv(torch.from_numpy(rois[:,1:5]), torch.from_numpy(bbox_pred)).numpy()
			boxes = _clip_boxes(boxes, blobs['im_info'][0,:2])

			# get scores of new regressed positions
			rois_score = Variable(torch.zeros(boxes.shape[0],1)).cuda()
			rois_bb = Variable(torch.from_numpy(boxes[:, cl*4:(cl+1)*4])).cuda()
			rois_pred = torch.cat((rois_score, rois_bb),1)
			_, scores, _, _ = self.frcnn.test_image(blobs['data'][0], blobs['im_info'], rois_pred)
			fc7 = self.frcnn.get_fc7()


			# check if still is a valid person
			inds = np.where(scores[:, cl] > 0.01)[0]
			self.pos2track = self.pos2track[inds]
			self.fc7 = self.fc7[inds]
			fc7 = fc7[inds]
			cls_boxes = boxes[inds, cl*4:(cl+1)*4]
			#self.cls_dets = np.hstack((cls_boxes, np.arange(cls_boxes.shape[0]+1,1,-1)[:, np.newaxis])) \
			#	.astype(np.float32, copy=False)
			#self.cls_dets = np.hstack((cls_boxes, scores[inds, cl][:, np.newaxis]+2)) \
			#	.astype(np.float32, copy=False)

			# compare fc7 features
			fc7_score = np.sum(np.abs(fc7-self.fc7),1)
			self.cls_dets = np.hstack((cls_boxes, fc7_score[:, np.newaxis]+2)) \
				.astype(np.float32, copy=False)
			

			# now do nms to check if tracks now fall together
			#keep = nms(torch.from_numpy(self.cls_dets), frcnn_cfg.TEST.NMS).numpy() if self.cls_dets.size > 0 else []
			keep = nms(torch.from_numpy(self.cls_dets), 0.7).numpy() if self.cls_dets.size > 0 else []
			keep = np.sort(keep)

			self.cls_dets = self.cls_dets[keep, :]
			self.pos2track = self.pos2track[keep]
			self.fc7 = self.fc7[keep]

		# number of old tracks
		old_tracks = self.cls_dets.shape[0]

		###########################
		# Look for new detections #
		###########################
		_, scores, bbox_pred, rois = self.frcnn.test_image(blobs['data'][0], blobs['im_info'])
		fc7 = self.frcnn.get_fc7()
		#print(fc7.shape)
		
		# Apply bounding-box regression deltas
		boxes = bbox_transform_inv(torch.from_numpy(rois[:,1:5]), torch.from_numpy(bbox_pred)).numpy()
		boxes = _clip_boxes(boxes, blobs['im_info'][0,:2])

		inds = np.where(scores[:, cl] > 0.05)[0]
		cls_scores = scores[inds, cl]
		cls_boxes = boxes[inds, cl*4:(cl+1)*4]
		fc7 = fc7[inds]

		cls_dets_new = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
			.astype(np.float32, copy=False)

		# nms to see if new detections are already taken care by prediction step
		self.cls_dets = np.vstack((self.cls_dets, cls_dets_new))  \
			.astype(np.float32, copy=False)


		keep = nms(torch.from_numpy(self.cls_dets), frcnn_cfg.TEST.NMS).numpy() if self.cls_dets.size > 0 else []
		keep = np.sort(keep)

		# only filter out new detections
		keep = np.hstack((np.arange(old_tracks), keep[keep>=old_tracks]))

		fc7 = fc7[keep[keep>=old_tracks]-old_tracks]
		#print(fc7.shape)

		#print(keep)
		self.cls_dets = self.cls_dets[keep, :]

		# construct new mapping pos 2 track for new detections
		new_tracks = self.cls_dets.shape[0] - old_tracks
		self.pos2track = np.hstack((self.pos2track, np.arange(self.track_num, self.track_num+new_tracks)))
		self.track_num += new_tracks


		for j,t in enumerate(self.cls_dets[:,0:4] / blobs['im_info'][0,2]):
			track_ind = self.pos2track[j]
			if track_ind not in self.results.keys():
				self.results[track_ind] = {}
				#self.results[track_ind]['fc7'] = fc7[j-old_tracks]
				self.fc7 = np.vstack((self.fc7, fc7[j-old_tracks]))
			self.results[track_ind][self.im_index] = t

		self.im_index += 1

		#print("{}\n{}".format(self.im_index-1,self.cls_dets))

	def get_results(self):
		return self.results
