import numpy as np
import torch
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment

from torchvision.ops.boxes import clip_boxes_to_image, nms

from .tracker import Tracker
from .utils import bbox_overlaps


class OracleTracker(Tracker):

	def __init__(self, obj_detect, reid_network, tracker_cfg, oracle_cfg):
		super(OracleTracker, self).__init__(obj_detect, reid_network, tracker_cfg)
		self.pos_oracle = oracle_cfg['pos_oracle']
		self.kill_oracle = oracle_cfg['kill_oracle']
		self.reid_oracle = oracle_cfg['reid_oracle']
		self.regress = oracle_cfg['regress']
		self.pos_oracle_center_only = oracle_cfg['pos_oracle_center_only']

	def tracks_to_inactive(self, tracks):
		super(OracleTracker, self).tracks_to_inactive(tracks)

		# only allow one track per GT in reid patience buffer
		if self.reid_oracle:
			inactive_tracks = []
			for t in reversed(self.inactive_tracks):
				if t.gt_id not in [t.gt_id for t in inactive_tracks]:
					inactive_tracks.append(t)
			self.inactive_tracks = inactive_tracks

	def add(self, new_det_pos, new_det_scores, new_det_features, blob):
		super(OracleTracker, self).add(
			new_det_pos, new_det_scores, new_det_features)

		num_new = new_det_pos.size(0)
		for t in self.tracks[-num_new:]:
			gt = blob['gt']
			boxes = torch.cat(list(gt.values()), 0).cuda()
			# boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data
			tracks_iou = bbox_overlaps(t.pos, boxes).cpu().numpy()
			ind = np.where(tracks_iou == np.max(tracks_iou))[1]
			if len(ind) > 0:
				ind = ind[0]
				overlap = tracks_iou[0, ind]
				if overlap >= 0.5:
					gt_id = list(gt.keys())[ind]
					t.gt_id = gt_id
					if self.pos_oracle:
						t.pos = gt[gt_id].cuda()

	def regress_tracks(self, blob):
		pos = self.get_pos()

		# regress
		boxes, scores = self.obj_detect.predict_boxes(pos)
		pos = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

		s = []
		for i in range(len(self.tracks) - 1, -1, -1):
			t = self.tracks[i]
			t.score = scores[i]

			if scores[i] <= self.regression_person_thresh and not self.kill_oracle:
				self.tracks_to_inactive([t])
			else:
				s.append(scores[i])
				if self.regress:
					t.pos = pos[i].view(1, -1)

		return torch.Tensor(s[::-1]).cuda()

	def reid(self, blob, new_det_pos, new_det_scores):
		new_det_features = [torch.zeros(0).cuda() for _ in range(len(new_det_pos))]

		if self.do_reid:
			new_det_features = self.reid_network.test_rois(
				blob['img'], new_det_pos).data

			if len(self.inactive_tracks) >= 1:
				# calculate appearance distances
				dist_mat, pos = [], []
				for t in self.inactive_tracks:
					dist_mat.append(torch.cat([t.test_features(feat.view(1, -1))
					                           for feat in new_det_features], dim=1))
					pos.append(t.pos)
				if len(dist_mat) > 1:
					dist_mat = torch.cat(dist_mat, 0)
					pos = torch.cat(pos, 0)
				else:
					dist_mat = dist_mat[0]
					pos = pos[0]

				# calculate IoU distances
				iou = bbox_overlaps(pos, new_det_pos)
				iou_mask = torch.ge(iou, self.reid_iou_threshold)
				iou_neg_mask = ~iou_mask
				# make all impossible assignments to the same add big value
				dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float() * 1000
				dist_mat = dist_mat.cpu().numpy()

				row_ind, col_ind = linear_sum_assignment(dist_mat)

				assigned = []
				remove_inactive = []
				for r,c in zip(row_ind, col_ind):
					if dist_mat[r,c] <= self.reid_sim_threshold:
						###### ADD GT ID ######
						gt = blob['gt']
						boxes = torch.cat(list(gt.values()), 0).cuda()
						# boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data
						tracks_iou = bbox_overlaps(t.pos, boxes).cpu().numpy()
						ind = np.where(tracks_iou==np.max(tracks_iou))[1]
						if len(ind) > 0:
							ind = ind[0]
							overlap = tracks_iou[0,ind]
							if overlap >= 0.5:
								gt_id = list(gt.keys())[ind]
								t.gt_id = gt_id
								if self.pos_oracle:
									t.pos = gt[gt_id].cuda()
							elif self.kill_oracle:
								continue
						t = self.inactive_tracks[r]
						self.tracks.append(t)
						t.count_inactive = 0
						t.reset_last_pos()
						t.pos = new_det_pos[c].view(1,-1)
						t.add_features(new_det_features[c].view(1,-1))
						assigned.append(c)
						remove_inactive.append(t)

				for t in remove_inactive:
					self.inactive_tracks.remove(t)

				keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
				if keep.nelement() > 0:
					new_det_pos = new_det_pos[keep]
					new_det_scores = new_det_scores[keep]
					new_det_features = new_det_features[keep]
				else:
					new_det_pos = torch.zeros(0).cuda()
					new_det_scores = torch.zeros(0).cuda()
					new_det_features = torch.zeros(0).cuda()

			if len(self.inactive_tracks) >= 1 and self.reid_oracle:
				gt = blob['gt']
				gt_pos = torch.cat(list(gt.values()), 0).cuda()
				gt_ids = list(gt.keys())

				# calculate IoU distances
				iou_neg = 1 - bbox_overlaps(new_det_pos, gt_pos)
				dist_mat = iou_neg.cpu().numpy()

				row_ind, col_ind = linear_sum_assignment(dist_mat)

				assigned = []
				for r, c in zip(row_ind, col_ind):
					if dist_mat[r, c] <= 0.5:
						gt_id = gt_ids[c]

						# loop through inactive in inversed order to get newest dead track
						for i in range(len(self.inactive_tracks) - 1, -1, -1):
							t = self.inactive_tracks[i]
							if t.gt_id == gt_id:
								if self.pos_oracle:
									t.pos = gt_pos[c].view(1, -1)
								else:
									t.pos = new_det_pos[r, :].view(1, -1)
								self.inactive_tracks.remove(t)
								self.tracks.append(t)
								t.reset_last_pos()
								assigned.append(r)

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

	def oracle(self, blob):
		gt = blob['gt']
		boxes = torch.cat(list(gt.values()), 0).cuda()
		ids = list(gt.keys())
		# boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).data

		# regress
		if self.pos_oracle:
			for t in self.tracks:
				if t.gt_id in gt.keys():
					new_pos = gt[t.gt_id].cuda()
					if self.pos_oracle_center_only:
						# extract center coordinates of track
						x1t = t.pos[0, 0]
						y1t = t.pos[0, 1]
						x2t = t.pos[0, 2]
						y2t = t.pos[0, 3]
						wt = x2t - x1t
						ht = y2t - y1t

						# extract coordinates of current pos
						new_pos = clip_boxes_to_image(new_pos, blob['img'].shape[-2:])
						x1n = new_pos[0, 0]
						y1n = new_pos[0, 1]
						x2n = new_pos[0, 2]
						y2n = new_pos[0, 3]
						cxn = (x2n + x1n) / 2
						cyn = (y2n + y1n) / 2

						# now set track to gt center coordinates
						t.pos[0, 0] = cxn - wt / 2
						t.pos[0, 1] = cyn - ht / 2
						t.pos[0, 2] = cxn + wt / 2
						t.pos[0, 3] = cyn + ht / 2
					else:
						t.pos = new_pos

		# now take care that all tracks are inside the image (normally done by regress)
		for t in self.tracks:
			t.pos = clip_boxes_to_image(t.pos, blob['img'].shape[-2:])

		if len(self.tracks):
			pos = self.get_pos()

			# calculate IoU distances
			iou_neg = 1 - bbox_overlaps(pos, boxes)
			dist_mat = iou_neg.cpu().numpy()

			row_ind, col_ind = linear_sum_assignment(dist_mat)

			matched = []
			# normal matching
			for r, c in zip(row_ind, col_ind):
				if dist_mat[r, c] <= 0.5:
					t = self.tracks[r]
					matched.append(t)
					t.gt_id = ids[c]

			if self.kill_oracle:
				self.tracks_to_inactive([t for t in self.tracks if t not in matched])

	def nms_oracle(self, blob, person_scores):
		gt = blob['gt']
		boxes = torch.cat(list(gt.values()), 0).cuda()
		ids = list(gt.keys())

		if len(self.tracks):
			pos = self.get_pos()

			# calculate IoU distances
			iou_neg = 1 - bbox_overlaps(pos, boxes)
			dist_mat = iou_neg.cpu().numpy()

			row_ind, col_ind = linear_sum_assignment(dist_mat)

			matched = []
			unmatched = []

			matched_index = []
			unmatched_index = []

			visibility = []
			visibility_index = []

			# check if tracks overlap and as soon as they do consider them a pair
			tracks_iou = bbox_overlaps(pos, pos).cpu().numpy()
			idx = np.where(tracks_iou >= self.regression_nms_thresh)

			tracks_ov = []
			for r, c in zip(idx[0], idx[1]):
				if r < c:
					tracks_ov.append([r, c])

			# check if overlapping tracks have GT matches
			for t0, t1 in tracks_ov:
				# get the matching gt indices
				gt_ids = []
				gt_pos = []
				gt_vis = []

				for i, t in enumerate([t0, t1]):
					ind = np.where(row_ind == t)[0]
					if len(ind) > 0:
						ind = ind[0]
						r = t
						c = col_ind[ind]
						if dist_mat[r, c] <= 0.5:
							gt_ids.append([ids[c], i])
							gt_pos.append(boxes[c].view(1, -1))
							gt_vis.append(blob['vis'][ids[c]])
						row_ind = np.delete(row_ind, ind)
						col_ind = np.delete(col_ind, ind)

				gt_ids = np.array(gt_ids)

				track0 = self.tracks[t0]
				track1 = self.tracks[t1]
				unm = [track0, track1]
				unm_index = [t0, t1]

				# any matches?
				if len(gt_ids) > 0:
					for t in list(unm):
						match = np.where(gt_ids[:, 0] == t.gt_id)[0]

						if len(match) > 0:
							unm.remove(t)
							matched.append(t)

							ind = self.tracks.index(t)
							matched_index.append(ind)
							unm_index.remove(ind)

				unmatched += unm
				unmatched_index += unm_index

				# if both are matched to a GT box one has to be killed by visibility
				if not len(unm):
					if gt_vis[0].gt(gt_vis[1]).all():
						visibility += [track1]
						visibility_index += [t1]
					else:
						visibility += [track0]
						visibility_index += [t0]

			# Remove unmatched NMS tracks
			for t in unmatched + visibility:
				if (t not in matched or t in visibility) and t in self.tracks:
					self.tracks.remove(t)
					self.inactive_tracks.append(t)

			index_remove = []
			for i in unmatched_index + visibility_index:
				if i not in matched_index or i in visibility_index:
					index_remove.append(i)

			keep = torch.Tensor([i for i in range(person_scores.size(0)) if i not in index_remove]).long().cuda()

			return person_scores[keep]

	def step(self, blob):
		for t in self.tracks:
			# add current position to last_pos list
			t.last_pos.append(t.pos.clone())

		###########################
		# Look for new detections #
		###########################

		self.obj_detect.load_image(blob['img'])

		if self.public_detections:
			dets = blob['dets'].squeeze(dim=0)
			if dets.nelement() > 0:
				boxes, scores = self.obj_detect.predict_boxes(dets)
			else:
				boxes = scores = torch.zeros(0).cuda()
		else:
			boxes, scores = self.obj_detect.detect(blob['img'])

		if boxes.nelement() > 0:
			boxes = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

			# Filter out tracks that have too low person score
			inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)

			if self.kill_oracle:
				gt = blob['gt']
				gt_boxes = torch.cat(list(gt.values()), 0).cuda()

				# calculate IoU distances
				iou_neg = 1 - bbox_overlaps(boxes, gt_boxes)
				dist_mat = iou_neg.cpu().numpy()

				row_ind, col_ind = linear_sum_assignment(dist_mat)

				matched = []
				# normal matching
				for r, c in zip(row_ind, col_ind):
					if dist_mat[r, c] <= 0.5:
						matched.append(r.item())

				inds = torch.LongTensor(matched).cuda()
			else:
				# Filter out tracks that have too low person score
				inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)
		else:
			inds = torch.zeros(0).cuda()

		if inds.nelement() > 0:
			det_pos = boxes[inds]

			det_scores = scores[inds]
		else:
			det_pos = torch.zeros(0).cuda()
			det_scores = torch.zeros(0).cuda()

		##################
		# Predict tracks #
		##################

		num_tracks = 0
		nms_inp_reg = torch.zeros(0).cuda()
		if len(self.tracks):
			# align
			if self.do_align:
				self.align(blob)
			if self.pos_oracle or self.kill_oracle:
				self.oracle(blob)
			elif self.motion_model_cfg['enabled']:
				self.motion()
				if self.reid_oracle:
					self.tracks_to_inactive([t for t in self.tracks if not t.has_positive_area()])
				else:
					self.tracks = [t for t in self.tracks if t.has_positive_area()]

			# regress
			if len(self.tracks):
				person_scores = self.regress_tracks(blob)
				# now NMS step
				if self.kill_oracle:
					person_scores = self.nms_oracle(blob, person_scores)

			if len(self.tracks):
				# create nms input


				# nms here if tracks overlap
				nms_inp_reg = torch.cat((self.get_pos(), person_scores.add_(3).view(-1, 1)), 1)
				if self.kill_oracle:
					# keep all
					keep = torch.arange(nms_inp_reg.size(0)).long().cuda()
				else:
					keep = nms(nms_inp_reg, self.regression_nms_thresh)

				# Plot the killed tracks for debugging
				self.tracks_to_inactive([self.tracks[i] for i in list(range(len(self.tracks))) if i not in keep])

				if keep.nelement() > 0 and self.do_reid:
					nms_inp_reg = nms_inp_reg[keep]
					new_features = self.get_appearances(blob)
					self.add_features(new_features)
					num_tracks = nms_inp_reg.size(0)

		#####################
		# Create new tracks #
		#####################

		if det_pos.nelement() > 0:
			keep = nms(det_pos, det_scores, self.detection_nms_thresh)
			det_pos = det_pos[keep]
			det_scores = det_scores[keep]

			# check with every track in a single run (problem if tracks delete each other)
			for t in self.tracks:
				nms_track_pos = torch.cat([t.pos, det_pos])
				nms_track_scores = torch.cat(
					[torch.tensor([2.0]).to(det_scores.device), det_scores])
				keep = nms(nms_track_pos, nms_track_scores, self.detection_nms_thresh)

				keep = keep[torch.ge(keep, 1)] - 1

				det_pos = det_pos[keep]
				det_scores = det_scores[keep]
				if keep.nelement() == 0:
					break

		if det_pos.nelement() > 0:
			new_det_pos = det_pos
			new_det_scores = det_scores

			# try to reidentify tracks
			new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

			# add new
			if new_det_pos.nelement() > 0:
				self.add(new_det_pos, new_det_scores, new_det_features, blob)

		####################
		# Generate Results #
		####################

		for t in self.tracks:
			if t.id not in self.results.keys():
				self.results[t.id] = {}
			self.results[t.id][self.im_index] = np.concatenate(
				[t.pos[0].cpu().numpy(), np.array([t.score])])

		for t in self.inactive_tracks:
			t.count_inactive += 1

		if not self.reid_oracle:
			self.inactive_tracks = [
				t for t in self.inactive_tracks if t.has_positive_area() and t.count_inactive <= self.inactive_patience
			]

		self.im_index += 1
		self.last_image = blob['img'][0]
