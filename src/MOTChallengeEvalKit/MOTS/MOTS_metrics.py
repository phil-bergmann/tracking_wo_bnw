import math
from collections import defaultdict
import pycocotools.mask as rletools
from mots_common.io import SegmentedObject
from mots_common.io import load_seqmap, load_sequences, load_txt
from Metrics import Metrics
import os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

# we only consider pedestrians
IGNORE_CLASS = 10
CLASS_ID = 2



def mask_iou(a, b, criterion="union"):
  is_crowd = criterion != "union"
  return rletools.iou([a.mask], [b.mask], [is_crowd])[0][0]




class MOTSMetrics(Metrics):
	def __init__(self, seqName = None):
		super().__init__()
		if seqName:
			self.seqName = seqName
		else: self.seqName = 0
		# Evaluation metrics

		self.register(name = "sMOTSA", formatter='{:.2f}'.format)
		self.register(name = "MOTSA", formatter='{:.2f}'.format)
		self.register(name = "MOTSP", formatter='{:.2f}'.format)
		self.register(name = "MOTSAL", formatter='{:.2f}'.format,  write_mail = False)
		self.register(name = "MODSA", formatter='{:.2f}'.format,  write_mail = False)
		self.register(name = "MODSP", formatter='{:.2f}'.format,  write_mail = False)


		self.register(name = "IDF1", formatter='{:.2f}'.format)
		self.register(name = "IDTP", formatter='{:.2f}'.format, write_mail = False)


		self.register(name = "MT", formatter='{:.0f}'.format)
		self.register(name = "PT", formatter='{:.0f}'.format, write_mail = False )
		self.register(name = "ML", formatter='{:.0f}'.format)

		self.register(name = "MTR", formatter='{:.2f}'.format)
		self.register(name = "PTR", formatter='{:.2f}'.format)
		self.register(name = "MLR", formatter='{:.2f}'.format)


		self.register(name = "n_gt_trajectories", display_name = "GT",formatter='{:.0f}'.format, write_mail = True)

		self.register(name = "tp", display_name="TP", formatter='{:.0f}'.format)  # number of true positives
		self.register(name = "fp", display_name="FP", formatter='{:.0f}'.format) # number of false positives
		self.register(name = "fn", display_name="FN", formatter='{:.0f}'.format)  # number of false negatives

		self.register(name = "recall", display_name="Rcll", formatter='{:.2f}'.format)
		self.register(name = "precision", display_name="Prcn", formatter='{:.2f}'.format)

		self.register(name = "F1", display_name="F1", formatter='{:.2f}'.format, write_mail = False)
		self.register(name = "FAR", formatter='{:.2f}'.format, write_mail = False)
		self.register(name = "total_cost", display_name="COST", formatter='{:.0f}'.format, write_mail = False)
		self.register(name = "fragments", display_name="FM", formatter='{:.0f}'.format)
		self.register(name = "fragments_rel", display_name="FMR", formatter='{:.2f}'.format)
		self.register(name = "id_switches", display_name="IDSW", formatter='{:.0f}'.format)

		self.register(name = "id_switches_rel", display_name="IDSWR", formatter='{:.1f}'.format)

		self.register(name = "n_tr_trajectories", display_name = "TR", formatter='{:.0f}'.format, write_mail = False)
		self.register(name = "total_num_frames", display_name="TOTAL_NUM", formatter='{:.0f}'.format, write_mail = False)


		self.register(name = "n_gt", display_name = "GT_OBJ", formatter='{:.0f}'.format, write_mail = False) # number of ground truth detections
		self.register(name = "n_tr", display_name = "TR_OBJ", formatter='{:.0f}'.format, write_mail = False) # number of tracker detections minus ignored tracker detections
		self.register(name = "n_itr",display_name="IGNORED",  formatter='{:.0f}'.format, write_mail = False)  # number of ignored tracker detections
		self.register(name = "id_n_tr", display_name = "ID_TR_OBJ", formatter='{:.0f}'.format, write_mail = False)
		self.register(name = "nbox_gt", display_name = "NBOX_GT", formatter='{:.0f}'.format, write_mail = False)




	
	def compute_clearmot(self):
	    # precision/recall etc.
	    if (self.fp + self.tp) == 0 or (self.tp + self.fn) == 0:
	        self.recall = 0.
	        self.precision = 0.
	    else:
	        self.recall = self.tp / float(self.tp + self.fn) * 100.
	        self.precision = self.tp / float(self.fp + self.tp) * 100.
	    if (self.recall + self.precision) == 0:
	        self.F1 = 0.
	    else:
	        self.F1 = (2. * (self.precision * self.recall) / (self.precision + self.recall) ) * 100.
	    if self.total_num_frames == 0:
	        self.FAR = "n/a"
	    else:
	        self.FAR = (self.fp / float(self.total_num_frames) ) 
	    # compute CLEARMOT
	    if self.n_gt == 0:
	        self.MOTSA = -float("inf")
	        self.MODSA = -float("inf")
	        self.sMOTSA = -float("inf")
	    else:
	        self.MOTSA = (1 - (self.fn + self.fp + self.id_switches) / float(self.n_gt) ) * 100.
	        self.MODSA = (1 - (self.fn + self.fp) / float(self.n_gt)) * 100.
	        self.sMOTSA = ((self.total_cost - self.fp - self.id_switches) / float(self.n_gt)) * 100.
	    if self.tp == 0:
	        self.MOTSP = float("inf")
	    else:
	        self.MOTSP = self.total_cost / float(self.tp) * 100.
	    if self.n_gt != 0:
	        if self.id_switches == 0:
	            self.MOTSAL = (1 - (self.fn + self.fp + self.id_switches) / float(self.n_gt)) * 100.
	        else:
	            self.MOTSAL = (1 - (self.fn + self.fp + math.log10(self.id_switches)) / float(
	            self.n_gt))*100.
	    else:
	        self.MOTSAL = -float("inf")
	
	    if self.total_num_frames == 0:
	        self.MODSP = "n/a"
	    else:
	        self.MODSP = self.MODSP / float(self.total_num_frames) * 100.




	    if self.n_gt_trajectories == 0:
	        self.MTR = 0.
	        self.PTR = 0.
	        self.MLR = 0.
	    else:
	        self.MTR = self.MT * 100. / float(self.n_gt_trajectories)
	        self.PTR = self.PT * 100. / float(self.n_gt_trajectories)
	        self.MLR = self.ML * 100. / float(self.n_gt_trajectories)


	    # calculate relative IDSW and FM

	    if self.recall != 0:
	        self.id_switches_rel = self.id_switches/self.recall*100
	        self.fragments_rel = self.fragments/self.recall* 100 

	    else:
	        self.id_switches_rel = float("inf")
	        self.fragments_rel = float("inf")




	    # IDF1
	    if self.n_gt_trajectories == 0:
	         self.IDF1 = 0.
	    else:
	         self.IDF1 = (2 * self.IDTP) / (self.nbox_gt + self.id_n_tr) * 100.
	    return self




	# go through all frames and associate ground truth and tracker results

	def compute_metrics_per_sequence(self, sequence, pred_file, gt_file, gtDataDir, benchmark_name,
						ignore_class = IGNORE_CLASS, class_id = CLASS_ID, overlap_function = mask_iou):

		gt_seq = load_txt(gt_file)
		results_seq = load_txt(pred_file)


		# load information about sequence
		import configparser
		config = configparser.ConfigParser()


		config.read(os.path.join(gtDataDir,  "seqinfo.ini"))

		max_frames = int(config['Sequence']["seqlength"])

		self.total_num_frames = max_frames + 1

		seq_trajectories = defaultdict(list)

		# To count number of track ids
		gt_track_ids = set()
		tr_track_ids = set()

		# Statistics over the current sequence
		seqtp = 0
		seqfn = 0
		seqfp = 0
		seqitr = 0

		n_gts = 0
		n_trs = 0

		frame_to_ignore_region = {}

		# Iterate over frames in this sequence
		for f in range(max_frames + 1):
			g = []
			dc = []
			t = []

			if f in gt_seq:
				for obj in gt_seq[f]:
					if obj.class_id == ignore_class:
						dc.append(obj)
					elif obj.class_id == class_id:
						g.append(obj)
						gt_track_ids.add(obj.track_id)
			if f in results_seq:
				for obj in results_seq[f]:
					if obj.class_id == class_id:
						t.append(obj)
						tr_track_ids.add(obj.track_id)

			# Handle ignore regions as one large ignore region
			dc = SegmentedObject(mask=rletools.merge([d.mask for d in dc], intersect=False),
			                                         class_id=ignore_class, track_id=ignore_class)
			frame_to_ignore_region[f] = dc

			tracks_valid = [False for _ in range(len(t))]

			# counting total number of ground truth and tracker objects
			self.n_gt += len(g)
			self.n_tr += len(t)

			n_gts += len(g)
			n_trs += len(t)

			# tmp variables for sanity checks and MODSP computation
			tmptp = 0
			tmpfp = 0
			tmpfn = 0
			tmpc = 0    # this will sum up the overlaps for all true positives
			tmpcs = [0] * len(g)    # this will save the overlaps for all true positives
			# the reason is that some true positives might be ignored
			# later such that the corrsponding overlaps can
			# be subtracted from tmpc for MODSP computation

			# To associate, simply take for each ground truth the (unique!) detection with IoU>0.5 if it exists

			# all ground truth trajectories are initially not associated
			# extend groundtruth trajectories lists (merge lists)
			for gg in g:
				seq_trajectories[gg.track_id].append(-1)
			num_associations = 0
			for row, gg in enumerate(g):
				for col, tt in enumerate(t):
					c = overlap_function(gg, tt)
					if c > 0.5:
						tracks_valid[col] = True
						self.total_cost += c
						tmpc += c
						tmpcs[row] = c
						seq_trajectories[g[row].track_id][-1] = t[col].track_id

						# true positives are only valid associations
						self.tp += 1
						tmptp += 1

						num_associations += 1

			# associate tracker and DontCare areas
			# ignore tracker in neighboring classes
			nignoredtracker = 0    # number of ignored tracker detections

			for i, tt in enumerate(t):
				overlap = overlap_function(tt, dc, "a")
				if overlap > 0.5 and not tracks_valid[i]:
					nignoredtracker += 1

			# count the number of ignored tracker objects
			self.n_itr += nignoredtracker

			# false negatives = non-associated gt instances
			#
			tmpfn += len(g) - num_associations
			self.fn += len(g) - num_associations

			# false positives = tracker instances - associated tracker instances
			# mismatches (mme_t)
			tmpfp += len(t) - tmptp - nignoredtracker
			self.fp += len(t) - tmptp - nignoredtracker
			# tmpfp     = len(t) - tmptp - nignoredtp # == len(t) - (tp - ignoredtp) - ignoredtp
			# self.fp += len(t) - tmptp - nignoredtp

			# update sequence data
			seqtp += tmptp
			seqfp += tmpfp
			seqfn += tmpfn
			seqitr += nignoredtracker

			# sanity checks
			# - the number of true positives minues ignored true positives
			#     should be greater or equal to 0
			# - the number of false negatives should be greater or equal to 0
			# - the number of false positives needs to be greater or equal to 0
			#     otherwise ignored detections might be counted double
			# - the number of counted true positives (plus ignored ones)
			#     and the number of counted false negatives (plus ignored ones)
			#     should match the total number of ground truth objects
			# - the number of counted true positives (plus ignored ones)
			#     and the number of counted false positives
			#     plus the number of ignored tracker detections should
			#     match the total number of tracker detections
			if tmptp < 0:
				print(tmptp)
				raise NameError("Something went wrong! TP is negative")
			if tmpfn < 0:
				print(tmpfn, len(g), num_associations)
				raise NameError("Something went wrong! FN is negative")
			if tmpfp < 0:
				print(tmpfp, len(t), tmptp, nignoredtracker)
				raise NameError("Something went wrong! FP is negative")
			if tmptp + tmpfn != len(g):
				print("seqname", seq_name)
				print("frame ", f)
				print("TP        ", tmptp)
				print("FN        ", tmpfn)
				print("FP        ", tmpfp)
				print("nGT     ", len(g))
				print("nAss    ", num_associations)
				raise NameError("Something went wrong! nGroundtruth is not TP+FN")
			if tmptp + tmpfp + nignoredtracker != len(t):
				print(seq_name, f, len(t), tmptp, tmpfp)
				print(num_associations)
				raise NameError("Something went wrong! nTracker is not TP+FP")

			# compute MODSP
			MODSP_f = 1
			if tmptp != 0:
				MODSP_f = tmpc / float(tmptp)
			self.MODSP += MODSP_f

		assert len(seq_trajectories) == len(gt_track_ids)
		self.n_gt_trajectories = len(gt_track_ids)
		self.n_tr_trajectories = len(tr_track_ids)

		# compute MT/PT/ML, fragments, idswitches for all groundtruth trajectories
		if len(seq_trajectories) != 0:
			for g in seq_trajectories.values():
				# all frames of this gt trajectory are not assigned to any detections
				if all([this == -1 for this in g]):
					self.ML += 1
					continue
				# compute tracked frames in trajectory
				last_id = g[0]
				# first detection (necessary to be in gt_trajectories) is always tracked
				tracked = 1 if g[0] >= 0 else 0
				for f in range(1, len(g)):
					if last_id != g[f] and last_id != -1 and g[f] != -1:
						self.id_switches += 1
					if f < len(g) - 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1 and g[f + 1] != -1:
						self.fragments += 1
					if g[f] != -1:
						tracked += 1
						last_id = g[f]
				# handle last frame; tracked state is handled in for loop (g[f]!=-1)
				if len(g) > 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1:
					self.fragments += 1

				# compute MT/PT/ML
				tracking_ratio = tracked / float(len(g))
				if tracking_ratio > 0.8:
					self.MT += 1
				elif tracking_ratio < 0.2:
					self.ML += 1
				else:    # 0.2 <= tracking_ratio <= 0.8
					self.PT += 1

			# compute IDF1
			idf1, idtp, nbox_gt, id_n_tr  = compute_idf1_and_idtp_for_sequence(gt_seq, results_seq, gt_track_ids, tr_track_ids, frame_to_ignore_region)
			self.IDTP = idtp
			#self.id_ign = id_ign
			self.id_n_tr = id_n_tr
			self.nbox_gt = nbox_gt
			
		return self



### IDF1 stuff
### code below adapted from https://github.com/shenh10/mot_evaluation/blob/5dd51e5cb7b45992774ea150e4386aa0b02b586f/utils/measurements.py
def compute_idf1_and_idtp_for_sequence(frame_to_gt, frame_to_pred, gt_ids, st_ids, frame_to_ignore_region):
	frame_to_can_be_ignored = {}
	for t in frame_to_pred.keys():
		preds_t = frame_to_pred[t]
		pred_masks_t = [p.mask for p in preds_t]
		ignore_region_t = frame_to_ignore_region[t].mask
		overlap = np.squeeze(rletools.iou(pred_masks_t, [ignore_region_t], [1]), axis=1)
		frame_to_can_be_ignored[t] = overlap > 0.5

	gt_ids = sorted(gt_ids)
	st_ids = sorted(st_ids)
	groundtruth = [[] for _ in gt_ids]
	prediction = [[] for _ in st_ids]
	for t, gts_t in frame_to_gt.items():
		for gt_t in gts_t:
			if gt_t.track_id in gt_ids:
				groundtruth[gt_ids.index(gt_t.track_id)].append((t, gt_t))
	for t in frame_to_pred.keys():
		preds_t = frame_to_pred[t]
		can_be_ignored_t = frame_to_can_be_ignored[t]
		assert len(preds_t) == len(can_be_ignored_t)
		for pred_t, ign_t in zip(preds_t, can_be_ignored_t):
			if pred_t.track_id in st_ids:
				prediction[st_ids.index(pred_t.track_id)].append((t, pred_t, ign_t))
	for gt in groundtruth:
		gt.sort(key=lambda x: x[0])
	for pred in prediction:
		pred.sort(key=lambda x: x[0])

	n_gt = len(gt_ids)
	n_st = len(st_ids)
	cost = np.zeros((n_gt + n_st, n_st + n_gt), dtype=float)
	cost[n_gt:, :n_st] = sys.maxsize    # float('inf')
	cost[:n_gt, n_st:] = sys.maxsize    # float('inf')

	fp = np.zeros(cost.shape)
	fn = np.zeros(cost.shape)
	ign = np.zeros(cost.shape)
	# cost matrix of all trajectory pairs
	cost_block, fp_block, fn_block, ign_block = cost_between_gt_pred(groundtruth, prediction)
	cost[:n_gt, :n_st] = cost_block
	fp[:n_gt, :n_st] = fp_block
	fn[:n_gt, :n_st] = fn_block
	ign[:n_gt, :n_st] = ign_block

	# computed trajectory match no groundtruth trajectory, FP
	for i in range(n_st):
		#cost[i + n_gt, i] = prediction[i].shape[0]
		#fp[i + n_gt, i] = prediction[i].shape[0]
		# don't count fp in case of ignore region
		fps = sum([~x[2] for x in prediction[i]])
		ig = sum([x[2] for x in prediction[i]])
		cost[i + n_gt, i] = fps
		fp[i + n_gt, i] = fps
		ign[i + n_gt, i] = ig

	# groundtruth trajectory match no computed trajectory, FN
	for i in range(n_gt):
		#cost[i, i + n_st] = groundtruth[i].shape[0]
		#fn[i, i + n_st] = groundtruth[i].shape[0]
		cost[i, i + n_st] = len(groundtruth[i])
		fn[i, i + n_st] = len(groundtruth[i])
	# TODO: add error handling here?
	matched_indices = linear_assignment(cost)
	#nbox_gt = sum([groundtruth[i].shape[0] for i in range(n_gt)])
	#nbox_st = sum([prediction[i].shape[0] for i in range(n_st)])
	nbox_gt = sum([len(groundtruth[i]) for i in range(n_gt)])

	nbox_st = sum([len(prediction[i]) for i in range(n_st)])

	#IDFP = 0
	IDFN = 0
	id_ign = 0
	for matched in zip(*matched_indices):
		#IDFP += fp[matched[0], matched[1]]
		IDFN += fn[matched[0], matched[1]]
		# exclude detections which are not matched and ignored from total count
		id_ign += ign[matched[0], matched[1]]
	id_n_tr = nbox_st - id_ign

	IDTP = nbox_gt - IDFN

	IDF1 = 2 * IDTP / (nbox_gt + id_n_tr)

	return IDF1, IDTP, nbox_gt, id_n_tr




def cost_between_gt_pred(groundtruth, prediction):
	n_gt = len(groundtruth)
	n_st = len(prediction)
	cost = np.zeros((n_gt, n_st), dtype=float)
	fp = np.zeros((n_gt, n_st), dtype=float)
	fn = np.zeros((n_gt, n_st), dtype=float)
	ign = np.zeros((n_gt, n_st), dtype=float)
	for i in range(n_gt):
		for j in range(n_st):
			fp[i, j], fn[i, j], ign[i, j] = cost_between_trajectories(groundtruth[i], prediction[j])
			cost[i, j] = fp[i, j] + fn[i, j]
	return cost, fp, fn, ign


def cost_between_trajectories(traj1, traj2):
	#[npoints1, dim1] = traj1.shape
	#[npoints2, dim2] = traj2.shape
	npoints1 = len(traj1)
	npoints2 = len(traj2)
	# find start and end frame of each trajectories
	#start1 = traj1[0, 0]
	#end1 = traj1[-1, 0]
	#start2 = traj2[0, 0]
	#end2 = traj2[-1, 0]
	times1 = [x[0] for x in traj1]
	times2 = [x[0] for x in traj2]
	start1 = min(times1)
	start2 = min(times2)
	end1 = max(times1)
	end2 = max(times2)

	ign = [traj2[i][2] for i in range(npoints2)]

	# check frame overlap
	#has_overlap = max(start1, start2) < min(end1, end2)
	# careful, changed this to <=, but I think now it's right
	has_overlap = max(start1, start2) <= min(end1, end2)
	if not has_overlap:
		fn = npoints1
		#fp = npoints2
		# disregard detections which can be ignored
		fp = sum([~x for x in ign])
		ig = sum(ign)
		return fp, fn, ig

	# gt trajectory mapping to st, check gt missed
	matched_pos1 = corresponding_frame(times1, npoints1, times2, npoints2)
	# st trajectory mapping to gt, check computed one false alarms
	matched_pos2 = corresponding_frame(times2, npoints2, times1, npoints1)
	overlap1 = compute_overlap(traj1, traj2, matched_pos1)
	overlap2 = compute_overlap(traj2, traj1, matched_pos2)
	# FN
	fn = sum([1 for i in range(npoints1) if overlap1[i] < 0.5])
	# FP
	# don't count false positive in case of ignore region
	unmatched = [overlap2[i] < 0.5 for i in range(npoints2)]
	#fp = sum([1 for i in range(npoints2) if overlap2[i] < 0.5 and not traj2[i][2]])
	fp = sum([1 for i in range(npoints2) if unmatched[i] and not ign[i]])
	ig = sum([1 for i in range(npoints2) if unmatched[i] and ign[i]])
	return fp, fn, ig


def corresponding_frame(traj1, len1, traj2, len2):
	"""
	Find the matching position in traj2 regarding to traj1
	Assume both trajectories in ascending frame ID
	"""
	p1, p2 = 0, 0
	loc = -1 * np.ones((len1,), dtype=int)
	while p1 < len1 and p2 < len2:
		if traj1[p1] < traj2[p2]:
			loc[p1] = -1
			p1 += 1
		elif traj1[p1] == traj2[p2]:
			loc[p1] = p2
			p1 += 1
			p2 += 1
		else:
			p2 += 1
	return loc


def compute_overlap(traj1, traj2, matched_pos):
	"""
	Compute the loss hit in traj2 regarding to traj1
	"""
	overlap = np.zeros((len(matched_pos),), dtype=float)
	for i in range(len(matched_pos)):
		if matched_pos[i] == -1:
			continue
		else:
			mask1 = traj1[i][1].mask
			mask2 = traj2[matched_pos[i]][1].mask
			iou = rletools.iou([mask1], [mask2], [False])[0][0]
			overlap[i] = iou
	return overlap

