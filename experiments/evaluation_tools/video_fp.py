from __future__ import absolute_import, division, print_function

import os
import os.path as osp
import time
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from cycler import cycler as cy
from torch.autograd import Variable
from torch.utils.data import DataLoader

import cv2
import seaborn as sns
from easydict import EasyDict as edict
from mot_evaluation.bbox import bbox_overlap
from mot_evaluation.io import (extract_valid_gt_data, print_metrics,
                               read_seqmaps, read_txt_to_struct)
from mot_evaluation.measurements import clear_mot_hungarian, idmeasures
from sacred import Experiment
from sklearn.utils.linear_assignment_ import linear_assignment
from tracktor.config import cfg, get_output_dir
from tracktor.datasets.factory import Datasets

sns.set_palette('deep')
sns.set(font_scale=1.5, rc={'text.usetex': True})


colors = ['aliceblue', 'black', 'green', 'red', 'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen',
'black']

colors = ['gray', 'black', 'green', 'red']


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
        assert not has_duplicates, 'Duplicate ID in same frame [Frame ID: %d].'%i
    keep_idx = np.where(res_keep == 1)[0]
    #print('[TRACK PREPROCESSING]: remove distractors and low visibility boxes, remaining %d/%d computed boxes'%(len(keep_idx), len(res_keep)))
    trackDB = trackDB[keep_idx, :]
    #print('Distractors:', distractor_ids)
    #keep_idx = np.array([i for i in xrange(gtDB.shape[0]) if gtDB[i, 1] not in distractor_ids and gtDB[i, 8] >= minvis])
    keep_idx = np.array([i for i in range(gtDB.shape[0]) if gtDB[i, 6] != 0])
    #print('[GT PREPROCESSING]: Removing distractor boxes, remaining %d/%d computed boxes'%(len(keep_idx), gtDB.shape[0]))
    gtDB = gtDB[keep_idx, :]
    return trackDB, gtDB


def evaluate_sequence(trackDB, gtDB, distractor_ids, iou_thres=0.5, minvis=0):
    """
    Evaluate single sequence
    trackDB: tracking result data structure
    gtDB: ground-truth data structure
    iou_thres: bounding box overlap threshold
    minvis: minimum tolerent visibility
    """
    trackDB, gtDB = preprocessingDB(trackDB, gtDB, distractor_ids, iou_thres, minvis)
    mme, c, fp, g, missed, d, M, allfps, clear_mot_info = clear_mot_hungarian(trackDB, gtDB, iou_thres)
    #print(mme)
    #print(c)
    #print(fp)
    #print(g)

    gt_frames = np.unique(gtDB[:, 0])
    gt_ids = np.unique(gtDB[:, 1])
    st_ids = np.unique(trackDB[:, 1])
    f_gt = len(gt_frames)
    n_gt = len(gt_ids)
    n_st = len(st_ids)

    FN = sum(missed)
    FP = sum(fp)
    IDS = sum(mme)
    MOTP = (sum(sum(d)) / sum(c)) * 100                                                 # MOTP = sum(iou) / # corrected boxes
    MOTAL = (1 - (sum(fp) + sum(missed) + np.log10(sum(mme) + 1)) / sum(g)) * 100       # MOTAL = 1 - (# fp + # fn + #log10(ids)) / # gts
    MOTA = (1 - (sum(fp) + sum(missed) + sum(mme)) / sum(g)) * 100                      # MOTA = 1 - (# fp + # fn + # ids) / # gts
    recall = sum(c) / sum(g) * 100                                                      # recall = TP / (TP + FN) = # corrected boxes / # gt boxes
    precision = sum(c) / (sum(fp) + sum(c)) * 100                                       # precision = TP / (TP + FP) = # corrected boxes / # det boxes
    FAR = sum(fp) / f_gt                                                                # FAR = sum(fp) / # frames
    MT_stats = np.zeros((n_gt, ), dtype=float)
    for i in range(n_gt):
        gt_in_person = np.where(gtDB[:, 1] == gt_ids[i])[0]
        gt_total_len = len(gt_in_person)
        gt_frames_tmp = gtDB[gt_in_person, 0].astype(int)
        gt_frames_list = list(gt_frames)
        st_total_len = sum([1 if i in M[gt_frames_list.index(f)].keys() else 0 for f in gt_frames_tmp])
        ratio = float(st_total_len) / gt_total_len

        if ratio < 0.2:
            MT_stats[i] = 1
        elif ratio >= 0.8:
            MT_stats[i] = 3
        else:
            MT_stats[i] = 2

    ML = len(np.where(MT_stats == 1)[0])
    PT = len(np.where(MT_stats == 2)[0])
    MT = len(np.where(MT_stats == 3)[0])

    # fragment
    fr = np.zeros((n_gt, ), dtype=int)
    M_arr = np.zeros((f_gt, n_gt), dtype=int)

    for i in range(f_gt):
        for gid in M[i].keys():
            M_arr[i, gid] = M[i][gid] + 1

    for i in range(n_gt):
        occur = np.where(M_arr[:, i] > 0)[0]
        occur = np.where(np.diff(occur) != 1)[0]
        fr[i] = len(occur)
    FRA = sum(fr)
    idmetrics = idmeasures(gtDB, trackDB, iou_thres)
    metrics = [idmetrics.IDF1, idmetrics.IDP, idmetrics.IDR, recall, precision, FAR, n_gt, MT, PT, ML, FP, FN, IDS, FRA, MOTA, MOTP, MOTAL]
    extra_info = edict()
    extra_info.mme = sum(mme)
    extra_info.c = sum(c)
    extra_info.fp = sum(fp)
    extra_info.g = sum(g)
    extra_info.missed = sum(missed)
    extra_info.d = d
    #extra_info.m = M
    extra_info.f_gt = f_gt
    extra_info.n_gt = n_gt
    extra_info.n_st = n_st
#    extra_info.allfps = allfps

    extra_info.ML = ML
    extra_info.PT = PT
    extra_info.MT = MT
    extra_info.FRA = FRA
    extra_info.idmetrics = idmetrics

    ML_PT_MT = [gt_ids[np.where(MT_stats == 1)[0]], gt_ids[np.where(MT_stats == 2)[0]], gt_ids[np.where(MT_stats == 3)[0]]]

    return metrics, extra_info, clear_mot_info, ML_PT_MT, M, gtDB, trackDB



def evaluate_bm(all_metrics):
    """
    Evaluate whole benchmark, summaries all metrics
    """
    f_gt, n_gt, n_st = 0, 0, 0
    nbox_gt, nbox_st = 0, 0
    c, g, fp, missed, ids = 0, 0, 0, 0, 0
    IDTP, IDFP, IDFN = 0, 0, 0
    MT, ML, PT, FRA = 0, 0, 0, 0
    overlap_sum = 0
    for i in range(len(all_metrics)):
        nbox_gt += all_metrics[i].idmetrics.nbox_gt
        nbox_st += all_metrics[i].idmetrics.nbox_st
        # Total ID Measures
        IDTP += all_metrics[i].idmetrics.IDTP
        IDFP += all_metrics[i].idmetrics.IDFP
        IDFN += all_metrics[i].idmetrics.IDFN
        # Total ID Measures
        MT += all_metrics[i].MT
        ML += all_metrics[i].ML
        PT += all_metrics[i].PT
        FRA += all_metrics[i].FRA
        f_gt += all_metrics[i].f_gt
        n_gt += all_metrics[i].n_gt
        n_st += all_metrics[i].n_st
        c += all_metrics[i].c
        g += all_metrics[i].g
        fp += all_metrics[i].fp
        missed += all_metrics[i].missed
        ids += all_metrics[i].mme
        overlap_sum += sum(sum(all_metrics[i].d))
    IDP = IDTP / (IDTP + IDFP) * 100                                # IDP = IDTP / (IDTP + IDFP)
    IDR = IDTP / (IDTP + IDFN) * 100                                # IDR = IDTP / (IDTP + IDFN)
    IDF1 = 2 * IDTP / (nbox_gt + nbox_st) * 100                     # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    FAR = fp /  f_gt
    MOTP = (overlap_sum / c) * 100
    MOTAL = (1 - (fp + missed + np.log10(ids + 1)) / g) * 100       # MOTAL = 1 - (# fp + # fn + #log10(ids)) / # gts
    MOTA = (1 - (fp + missed + ids) / g) * 100                      # MOTA = 1 - (# fp + # fn + # ids) / # gts
    recall = c / g * 100                                            # recall = TP / (TP + FN) = # corrected boxes / # gt boxes
    precision = c / (fp + c) * 100                                  # precision = TP / (TP + FP) = # corrected boxes / # det boxes
    metrics = [IDF1, IDP, IDR, recall, precision, FAR, n_gt, MT, PT, ML, fp, missed, ids, FRA, MOTA, MOTP, MOTAL]
    return metrics

def evaluate_tracking(sequences, track_dir, gt_dir):
    all_info = []
    for seqname in sequences:
        track_res = os.path.join(track_dir, seqname, 'res.txt')
        gt_file = os.path.join(gt_dir, seqname, 'gt.txt')
        assert os.path.exists(track_res) and os.path.exists(gt_file), 'Either tracking result or groundtruth directory does not exist'

        trackDB = read_txt_to_struct(track_res)
        gtDB = read_txt_to_struct(gt_file)

        gtDB, distractor_ids = extract_valid_gt_data(gtDB)
        metrics, extra_info = evaluate_sequence(trackDB, gtDB, distractor_ids)
        print_metrics(seqname + ' Evaluation', metrics)
        all_info.append(extra_info)
    all_metrics = evaluate_bm(all_info)
    print_metrics('Summary Evaluation', all_metrics)

def evaluate_new(stDB, gtDB, distractor_ids):

    #trackDB = read_txt_to_struct(results)
    #gtDB = read_txt_to_struct(gt_file)

    #gtDB, distractor_ids = extract_valid_gt_data(gtDB)

    metrics, extra_info, clear_mot_info, ML_PT_MT, M, gtDB, stDB = evaluate_sequence(stDB, gtDB, distractor_ids)

    #print_metrics(' Evaluation', metrics)

    return clear_mot_info, M, gtDB, stDB

@ex.automain
def my_main(_config):

    print(_config)

    dataset = "mot_train_"
    detections = "FRCNN"

    ##########################
    # Initialize the modules #
    ##########################

    print("[*] Beginning evaluation...")
    module_dir = get_output_dir('MOT17')
    results_dir = module_dir
    module_dir = osp.join(module_dir, 'eval/video_fp')
    #output_dir = osp.join(results_dir, 'plots')
    #if not osp.exists(output_dir):
    #    os.makedirs(output_dir)

    #sequences_raw = ["MOT17-13", "MOT17-11", "MOT17-10", "MOT17-09", "MOT17-05", "MOT17-04", "MOT17-02", ]

    #sequences = ["{}-{}".format(s, detections) for s in sequences_raw]
    #sequences = sequences[:1]

    # tracker = ["FRCNN_Base", "HAM_SADF17", "MOTDT17", "EDMT17", "IOU17", "MHT_bLSTM", "FWT_17", "jCC", "MHT_DAM_17"]
    # tracker = ["Baseline", "BnW", "FWT_17", "jCC", "MOTDT17", "MHT_DAM_17"]
    tracker = ["Tracktor", "FWT", "jCC", "MOTDT17"]
    #tracker = ["Baseline"]

    for t in tracker:
        print("[*] Evaluating {}".format(t))
        if True:
        #for db in Datasets(dataset):
            ################################
            # Make videos for each tracker #
            ################################
            db = Datasets(dataset)[2]

            s = "{}-{}".format(db, detections)

            gt_file = osp.join(cfg.DATA_DIR, "MOT17Labels", "train", s, "gt", "gt.txt")
            res_file = osp.join(results_dir, t, s+".txt")

            stDB = read_txt_to_struct(res_file)
            gtDB = read_txt_to_struct(gt_file)

            gtDB, distractor_ids = extract_valid_gt_data(gtDB)
            _, M, gtDB, stDB = evaluate_new(stDB, gtDB, distractor_ids)
            #gt_ids_res = np.unique(gtDB[:, 1])

            #gtDB = read_txt_to_struct(gt_file)
            # filter out so that confidence and id = 1
            #gtDB = gtDB[gtDB[:,7] == 1]
            #gtDB = gtDB[gtDB[:,6] == 1]

            st_ids = np.unique(stDB[:, 1])
            #gt_ids = np.unique(gtDB[:, 1])
            gt_frames = np.unique(gtDB[:, 0])
            f_gt = len(gt_frames)

            #gt_inds = [{} for i in range(f_gt)]
            st_inds = [{} for i in range(f_gt)]

            # hash the indices to speed up indexing
            #for i in range(gtDB.shape[0]):
            #    frame = np.where(gt_frames == gtDB[i, 0])[0][0]
                #gid = np.where(gt_ids == gtDB[i, 1])[0][0]
            #    gt_id = int(gtDB[i,1])
            #    gt_inds[frame][gt_id] = i

            gt_frames_list = list(gt_frames)
            for i in range(stDB.shape[0]):
                # sometimes detection missed in certain frames, thus should be assigned to groundtruth frame id for alignment
                frame = gt_frames_list.index(stDB[i, 0])
                sid = np.where(st_ids == stDB[i, 1])[0][0]
                st_inds[frame][sid] = i

            #stDB = read_txt_to_struct(res_file)

            results = []
            for frame in range(f_gt):
                # get gt_ids in res
                m = M[frame]
                matched_sids = list(m.values())

                #frame_sids = list(st_inds[frame].keys())

                f = gt_frames_list[frame]
                st_frame = stDB[stDB[:,0] == f]
                st_uniq_ids = np.unique(st_frame[:,1])

                for st_id in st_uniq_ids:
                    sid = -1
                    si = np.where(st_ids == st_id)[0]
                    if len(si) > 0:
                        sid = si[0]
                    if sid not in matched_sids:
                        res = np.zeros(6)
                        res[0] = frame+1
                        st_track = st_frame[st_frame[:,1]==st_id]
                        res[2:6] = st_track[0,2:6]
                        results.append(res)
                    else:
                        matched_sids.remove(sid)

            results = np.array(results)
            print(results.shape[0])

            output_dir = osp.join(module_dir, t, s)
            if not osp.exists(output_dir):
                os.makedirs(output_dir)

            print("[*] Plotting whole sequence to {}".format(output_dir))

            # infinte color loop
            cyl = cy('ec', colors)
            loop_cy_iter = cyl()
            styles = defaultdict(lambda : next(loop_cy_iter))

            for frame,v in enumerate(db,1):
                im_path = v['im_path']
                im_name = osp.basename(im_path)
                im_output = osp.join(output_dir, im_name)
                im = cv2.imread(im_path)
                im = im[:, :, (2, 1, 0)]

                sizes = np.shape(im)
                height = float(sizes[0])
                width = float(sizes[1])

                fig = plt.figure()
                #fig.set_size_inches(w,h)
                #fig.set_size_inches(width/height, 1, forward=False)
                #fig.set_size_inches(width/100, height/100)
                scale = width / 640
                #fig.set_size_inches(640/100, height*scale/100)
                fig.set_size_inches(width / 100, height / 100)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(im)

                res_frame = results[results[:,0]==frame]

                for j in range(res_frame.shape[0]):
                    box = res_frame[j,2:6]
                    gt_id = int(res_frame[j,1])
                    ax.add_patch(
                        plt.Rectangle((box[0], box[1]),
                            box[2] - box[0],
                            box[3] - box[1], fill=False,
                            linewidth=1.3*scale, color='blue')
                    )

                ax.annotate(t, (width - 250, height - 100),
                            color='white', weight='bold', fontsize=72, ha='center', va='center')

                plt.axis('off')
                plt.draw()
                plt.savefig(im_output, dpi=100)
                plt.close()
