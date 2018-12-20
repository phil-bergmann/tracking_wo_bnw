from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from sacred import Experiment
from model.config import cfg as frcnn_cfg
from model.config import cfg_from_list, cfg_from_file
import os
import os.path as osp
import yaml
import time
import csv
from scipy.optimize import linear_sum_assignment

from tracker.rfrcnn import FRCNN as rFRCNN
from tracker.vfrcnn import FRCNN as vFRCNN
from tracker.config import cfg, get_output_dir
from tracker.utils import plot_sequence
from tracker.datasets.factory import Datasets
from tracker.tracker import Tracker
from tracker.utils import interpolate, bbox_overlaps, plot_sequence
from tracker.resnet import resnet50

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from mot_evaluation.io import read_txt_to_struct

ex = Experiment()

#sequences_raw = ["MOT17-13", "MOT17-11", "MOT17-10", "MOT17-09", "MOT17-05", "MOT17-04", "MOT17-02", ]
#sequences = ["{}-{}".format(s, detections) for s in sequences_raw]
dataset = "mot_train_"

@ex.automain
def my_main(_config):

    dets = "FRCNN"
    plot = True

    print(_config)

    output_dir = osp.join(get_output_dir("MOT_analysis"), "DET_GT_INTERPOL")
    
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    ##########################
    # Initialize the modules #
    ##########################

    print("[*] Beginning evaluation...")

    for db in Datasets(dataset):
        s = "{}-{}".format(db, dets)

        print("[*] Evaluating: {}".format(s))

        gt_file = osp.join(cfg.DATA_DIR, "MOT17Labels", "train", s, "gt", "gt.txt")
        det_file = osp.join(cfg.DATA_DIR, "MOT17Labels", "train", s, "det", "det.txt")

        dtDB = read_txt_to_struct(det_file)
        gtDB = read_txt_to_struct(gt_file)

        # filter out so that confidence and id = 1
        gtDB = gtDB[gtDB[:,7] == 1]
        gtDB = gtDB[gtDB[:,6] == 1]

        #tracker.write_debug(osp.join(output_dir, "debug_{}.txt".format(s)))

        results = {}

        for frame in np.unique(gtDB[:, 0]):
            detections = dtDB[dtDB[:,0] == frame]
            det_boxes = detections[:,2:6]

            groundtruth = gtDB[gtDB[:,0] == frame]
            gt_boxes = groundtruth[:,2:6]

            if len(det_boxes) == 0 or len(gt_boxes) == 0:
                continue

            # calculate IoU distances
            iou = bbox_overlaps(det_boxes, gt_boxes)

            row_ind, col_ind = linear_sum_assignment(1 - iou)

            for r,c in zip(row_ind, col_ind):
                if iou[r,c] >= 0.5:
                    gt_id = groundtruth[c,1]

                    if gt_id not in results.keys():
                        results[gt_id] = {}
                    results[gt_id][int(frame)-1] = det_boxes[r]

        for gt_id, track in results.items():
            groundtruth = gtDB[gtDB[:,1] == gt_id]

            keys = track.keys()

            for f in range(min(keys)+1, max(keys)):
                if f in keys:
                    continue
                box = groundtruth[groundtruth[:,0] == f+1, 2:6][0]
                results[gt_id][f] = box


        #results = interpolate(results)

        res_file = osp.join(output_dir, s+".txt")

        print("[*] Writing to: {}".format(res_file))

        with open(res_file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in results.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow([frame+1, i, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])

        if plot:
            plot_sequence(results, db, osp.join(output_dir, s))
