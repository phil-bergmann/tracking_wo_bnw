from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from sacred import Experiment
from model.config import cfg as frcnn_cfg
import os
import os.path as osp
import yaml
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt

from tracker.rfrcnn import FRCNN as rFRCNN
from tracker.vfrcnn import FRCNN as vFRCNN
from tracker.config import cfg, get_output_dir
from tracker.utils import plot_sequence
from tracker.mot_sequence import MOT_Sequence
from tracker.kitti_sequence import KITTI_Sequence
from tracker.tracker_debug import Tracker
from tracker.utils import interpolate
from tracker.resnet import resnet50

from sklearn.utils.linear_assignment_ import linear_assignment
from easydict import EasyDict as edict
from mot_evaluation.io import read_txt_to_struct, read_seqmaps, extract_valid_gt_data, print_metrics
from mot_evaluation.bbox import bbox_overlap
from mot_evaluation.measurements import clear_mot_hungarian, idmeasures

ex = Experiment()

def preprocessingDB(trackDB, gtDB, distractor_ids, iou_thres, minvis):
    """
    Preprocess the computed trajectory data.
    Matching computed boxes to groundtruth to remove distractors and low visibility data in both trackDB and gtDB
    trackDB: [npoints, 9] computed trajectory data
    gtDB: [npoints, 9] computed trajectory data
    distractor_ids: identities of distractors of the sequence
    iou_thres: bounding box overlap threshold
    minvis: minimum visibility of groundtruth boxes, default set to zero because the occluded people are supposed to be interpolated for tracking.
    """
    track_frames = np.unique(trackDB[:, 0])
    gt_frames = np.unique(gtDB[:, 0])
    nframes = min(len(track_frames), len(gt_frames))  
    res_keep = np.ones((trackDB.shape[0], ), dtype=float)
    for i in range(1, nframes + 1):
        # find all result boxes in this frame
        res_in_frame = np.where(trackDB[:, 0] == i)[0]
        res_in_frame_data = trackDB[res_in_frame, :]
        gt_in_frame = np.where(gtDB[:, 0] == i)[0]
        gt_in_frame_data = gtDB[gt_in_frame, :]
        res_num = res_in_frame.shape[0]
        gt_num = gt_in_frame.shape[0]
        overlaps = np.zeros((res_num, gt_num), dtype=float)
        for gid in range(gt_num):
            overlaps[:, gid] = bbox_overlap(res_in_frame_data[:, 2:6], gt_in_frame_data[gid, 2:6]) 
        matched_indices = linear_assignment(1 - overlaps)
        for matched in matched_indices:
            # overlap lower than threshold, discard the pair
            if overlaps[matched[0], matched[1]] < iou_thres:
                continue

            # matched to distractors, discard the result box
            if gt_in_frame_data[matched[1], 1] in distractor_ids:
                res_keep[res_in_frame[matched[0]]] = 0
            
            # matched to a partial
            if gt_in_frame_data[matched[1], 8] < minvis:
                res_keep[res_in_frame[matched[0]]] = 0
            

        # sanity check
        frame_id_pairs = res_in_frame_data[:, :2]
        uniq_frame_id_pairs = np.unique(frame_id_pairs)
        has_duplicates = uniq_frame_id_pairs.shape[0] < frame_id_pairs.shape[0]
        #assert not has_duplicates, 'Duplicate ID in same frame [Frame ID: %d].'%i
    keep_idx = np.where(res_keep == 1)[0]
    print('[TRACK PREPROCESSING]: remove distractors and low visibility boxes, remaining %d/%d computed boxes'%(len(keep_idx), len(res_keep)))
    trackDB = trackDB[keep_idx, :]
    print('Distractors:', distractor_ids)
    #keep_idx = np.array([i for i in xrange(gtDB.shape[0]) if gtDB[i, 1] not in distractor_ids and gtDB[i, 8] >= minvis])
    keep_idx = np.array([i for i in range(gtDB.shape[0]) if gtDB[i, 6] != 0] )
    print('[GT PREPROCESSING]: Removing distractor boxes, remaining %d/%d computed boxes'%(len(keep_idx), gtDB.shape[0]))
    gtDB = gtDB[keep_idx, :]
    return trackDB, gtDB

@ex.automain
def my_main(_config):

    print(_config)

    ##########################
    # Initialize the modules #
    ##########################
    
    print("[*] Beginning evaluation...")
    output_dir = osp.join(get_output_dir('MOT_analysis'), 'coverage')

    sequences = ["MOT17-13-DPM", "MOT17-11-DPM", "MOT17-10-DPM", "MOT17-09-DPM", "MOT17-05-DPM", "MOT17-04-DPM", "MOT17-02-DPM", ]
    tracker = ["FRCNN", "FWT_17", "jCC", "MHT_DAM_17", "SST"]
    tracker = ["FRCNN"]

    for t in tracker:
        data_points = []
        for s in sequences:
            ########################################
            # Get DPM / GT coverage for each track #
            ########################################

            gt_file = osp.join(cfg.DATA_DIR, "MOT17Labels", "train", s, "gt", "gt.txt")
            dpm_file = osp.join(cfg.DATA_DIR, "MOT17Labels", "train", s, "det", "det.txt")

            gtDB = read_txt_to_struct(gt_file)
            dpmDB = read_txt_to_struct(dpm_file)
            gtDB, distractor_ids = extract_valid_gt_data(gtDB)
            dpmDB, gtDB = preprocessingDB(dpmDB, gtDB, distractor_ids, 0.5, 0)

            gt_ids = np.unique(gtDB[:, 1])

            gt_ges = {int(i):0 for i in gt_ids}
            gt_matched = {int(i):0 for i in gt_ids}
            gt_tracked = {int(i):0 for i in gt_ids}

            track_frames = np.unique(dpmDB[:, 0])
            gt_frames = np.unique(gtDB[:, 0])
            nframes = min(len(track_frames), len(gt_frames))
            res_keep = np.ones((dpmDB.shape[0], ), dtype=float)
            for i in range(1, nframes + 1):
                # find all result boxes in this frame
                res_in_frame = np.where(dpmDB[:, 0] == i)[0]
                res_in_frame_data = dpmDB[res_in_frame, :]
                gt_in_frame = np.where(gtDB[:, 0] == i)[0]
                gt_in_frame_data = gtDB[gt_in_frame, :]

                #for gt in gt_in_frame_data:
                #    gt_ges[int(gt[1])] += 1

                res_num = res_in_frame.shape[0]
                gt_num = gt_in_frame.shape[0]
                overlaps = np.zeros((res_num, gt_num), dtype=float)
                for gid in range(gt_num):
                    overlaps[:, gid] = bbox_overlap(res_in_frame_data[:, 2:6], gt_in_frame_data[gid, 2:6])
                matched_indices = linear_assignment(1 - overlaps)
                for matched in matched_indices:
                    # overlap lower than threshold, discard the pair
                    if overlaps[matched[0], matched[1]] > 0.5:
                        gt_id = int(gt_in_frame_data[matched[1], 1])
                        gt_matched[gt_id] += 1

            for k in gt_ids:
                gt_ges[k] = len(np.where(gtDB[:, 1] == k)[0])
                gt_tracked[k] = gt_matched[k] / gt_ges[k]

            res_file = osp.join(output_dir, t, s+".txt")

            gtDB = read_txt_to_struct(gt_file)
            trackDB = read_txt_to_struct(res_file)
            gtDB, distractor_ids = extract_valid_gt_data(gtDB)
            trackDB, gtDB = preprocessingDB(trackDB, gtDB, distractor_ids, 0.5, 0)

            tr_matched = {int(i):0 for i in gt_ids}
            tr_tracked = {int(i):0 for i in gt_ids}

            track_frames = np.unique(trackDB[:, 0])
            gt_frames = np.unique(gtDB[:, 0])
            nframes = min(len(track_frames), len(gt_frames))
            res_keep = np.ones((trackDB.shape[0], ), dtype=float)
            for i in range(1, nframes + 1):
                # find all result boxes in this frame
                res_in_frame = np.where(trackDB[:, 0] == i)[0]
                res_in_frame_data = trackDB[res_in_frame, :]
                gt_in_frame = np.where(gtDB[:, 0] == i)[0]
                gt_in_frame_data = gtDB[gt_in_frame, :]

                res_num = res_in_frame.shape[0]
                gt_num = gt_in_frame.shape[0]
                overlaps = np.zeros((res_num, gt_num), dtype=float)
                for gid in range(gt_num):
                    overlaps[:, gid] = bbox_overlap(res_in_frame_data[:, 2:6], gt_in_frame_data[gid, 2:6])
                matched_indices = linear_assignment(1 - overlaps)
                for matched in matched_indices:
                    # overlap lower than threshold, discard the pair
                    if overlaps[matched[0], matched[1]] > 0.5:
                        gt_id = int(gt_in_frame_data[matched[1], 1])
                        tr_matched[gt_id] += 1

            for k in gt_ids:
                data_points.append([gt_tracked[k], tr_matched[k] / gt_ges[k]])

        data_points = np.array(data_points)
        plt.plot([0,1], [0,1], 'r-')
        plt.scatter(data_points[:,0], data_points[:,1], s=2**2)
        plt.xlabel('DPM coverage')
        plt.ylabel('tracker coverage')
        plt.savefig(osp.join(output_dir, t+".eps"), format='eps')
        plt.close()
