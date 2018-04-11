from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms
from model.config import cfg as frcnn_cfg

import torch
from torch.autograd import Variable
import numpy as np
import cv2
from os.path import join

class Simple_Flow_Tracker():

	def __init__(self, frcnn, flownet, detection_person_thresh, regression_person_thresh, detection_nms_thresh,
		regression_nms_thresh, alive_patience, use_detections):
		self.frcnn = frcnn
		self.flownet = flownet
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
			self.prev_image = None

	def keep(self, keep):
		self.pos = self.pos[keep]
		#self.features = self.features[keep]
		self.ind2track = self.ind2track[keep]
		self.kill_counter = self.kill_counter[keep]

	def step(self, blob):

		cl = 1

		###########################
		# Look for new detections #
		###########################
		_, scores, bbox_pred, rois = self.frcnn.test_image(blob['data'][0], blob['im_info'][0])
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
		nms_inp_reg = self.pos.new(0)
		if self.pos.nelement() > 0:
			# calculate flow
			if self.im_index > 0:
				# input has to be multiple of 64
				# format of images NxHxWxC, transpose(3,0,1,2)
				# input format for flownet: 5d tensor, (batch, rgb, img #, x, y)
				im0 = self.prev_image.cpu().numpy()[0]
				im1 = blob['or_data'][0].cpu().numpy()[0]
				image_h = im1.shape[0]
				image_w = im1.shape[1]
				crop_h = (im1.shape[0] // 64) * 64
				crop_w = (im1.shape[1] // 64) * 64


				im0 = cv2.resize(im0, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
				im1 = cv2.resize(im1, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
				im0 = torch.from_numpy(im0).unsqueeze(0).cuda()
				im1 = torch.from_numpy(im1).unsqueeze(0).cuda()
				"""
				im0 = self.prev_image.cuda()
				im1 = blob['data'][0].cuda()
				image_h = (im1.size(1) // 64) * 64
				image_w = (im1.size(2) // 64) * 64
				crop = StaticCenterCrop([im1.size(1), im1.size(2)], [image_h, image_w])
				im0 = crop(im0)
				im1 = crop(im1)
				"""
				# BGR to RGB
				im0[:,:,:,:] = im0[:,:,:,[2,1,0]]
				im1[:,:,:,:] = im1[:,:,:,[2,1,0]]
				images = torch.cat((im0, im1), 0).permute(3,0,1,2).unsqueeze(0)
				flow = self.flownet(Variable(images, volatile = True))
				flow = flow.data[0].permute(1, 2, 0)
				#flow = flow[0].data.cpu().numpy().transpose(1, 2, 0)
				pos = self.pos / blob['im_info'][0][2]
				fl = torch.stack([flow[int(p[1]):int(p[3])+1, int(p[0]):int(p[2])+1, :].mean(dim=0).mean(dim=0) for p in pos], 0)
				self.pos[:,0] += fl[:,0] * 8
				self.pos[:,1] += fl[:,1] * 8
				self.pos[:,2] += fl[:,0] * 8
				self.pos[:,3] += fl[:,1] * 8
				#writeFlow(join('/usr/stud/bergmanp/test_flow', "{:06d}.flo".format(self.im_index)), fl)
			# regress
			_, _, bbox_pred, rois = self.frcnn.test_rois(self.pos)
			boxes = bbox_transform_inv(rois, bbox_pred)
			boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data
			#boxes = clip_boxes(Variable(rois.contiguous()), blob['im_info'][0][:2]).data
			self.pos = boxes[:,cl*4:(cl+1)*4]
			#self.pos = boxes

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
				nms_inp_reg = torch.cat((self.pos, scores.add_(2).view(-1,1)),1)

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

		self.im_index += 1
		self.prev_image = blob['or_data'][0]

		#print("tracks active: {}/{}".format(num_tracks, self.track_num))

	def get_results(self):
		return self.results

class StaticCenterCrop(object):
	def __init__(self, image_size, crop_size):
		self.th, self.tw = crop_size
		self.h, self.w = image_size
	def __call__(self, img):
		return img[:,int((self.h-self.th)/2):int((self.h+self.th)/2), int((self.w-self.tw)/2):int((self.w+self.tw)/2),:]

def writeFlow(filename,uv,v=None):
	""" Write optical flow to file.
	
	If v is None, uv is assumed to contain both u and v channels,
	stacked in depth.
	Original code by Deqing Sun, adapted from Daniel Scharstein.
	"""
	TAG_CHAR = np.array([202021.25], np.float32)
	
	nBands = 2

	if v is None:
		assert(uv.ndim == 3)
		assert(uv.shape[2] == 2)
		u = uv[:,:,0]
		v = uv[:,:,1]
	else:
		u = uv

	assert(u.shape == v.shape)
	height,width = u.shape
	f = open(filename,'wb')
	# write the header
	f.write(TAG_CHAR)
	np.array(width).astype(np.int32).tofile(f)
	np.array(height).astype(np.int32).tofile(f)
	# arrange into matrix form
	tmp = np.zeros((height, width*nBands))
	tmp[:,np.arange(width)*2] = u
	tmp[:,np.arange(width)*2 + 1] = v
	tmp.astype(np.float32).tofile(f)
	f.close()
