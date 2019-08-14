from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sacred import Experiment
import os
import os.path as osp
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('deep')
sns.set(font_scale=1.5, rc={'text.usetex' : True})

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

    return metrics, extra_info, clear_mot_info, ML_PT_MT, M, gtDB



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

    metrics, extra_info, clear_mot_info, ML_PT_MT, M, gtDB = evaluate_sequence(stDB, gtDB, distractor_ids)

    #print_metrics(' Evaluation', metrics)

    return clear_mot_info, M, gtDB

@ex.automain
def my_main(_config):

    print(_config)

    ##########################
    # Initialize the modules #
    ##########################

    print("[*] Beginning evaluation...")

    ###################### PFAD ANPASSEN ###################

    results_dir = osp.join('output/tracktor/MOT17')

    ########################################################

    output_dir = osp.join(results_dir, 'eval/track_missed_found')
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    sequences_raw = ["MOT17-13", "MOT17-11", "MOT17-10", "MOT17-09", "MOT17-05", "MOT17-04", "MOT17-02", ]

    ######### HIER DETECTIONS ÄNDERNS (DPM, FRCNN, SDP) #

    detections = "SDP"

    #####################################################


    sequences = ["{}-{}".format(s, detections) for s in sequences_raw]
    #sequences = sequences[:1]

    #tracker = ["FRCNN_Base", "HAM_SADF17", "MOTDT17", "EDMT17", "IOU17", "MHT_bLSTM", "FWT_17", "jCC", "MHT_DAM_17"]
    tracker = ["Tracktor", "Tracktor++", "FWT", "jCC", "MOTDT17", "MHT_DAM"]
    #tracker = tracker[:2]
    # "PHD_GSDL17" does not work, error
    #tracker = tracker[-4:]

    poly_missed_gaps_length = []
    tv = []
    th = []
    gt_vis = []
    gt_h = []
    for t in tracker:
        print("[*] Evaluating {}".format(t))
        missed_gaps_length = []
        missed_visibilities = []
        tracked_visibilities = []
        tracked_visibilities_heights = []
        missed_visibilities_heights = []
        for s in sequences:
            ########################################
            # Get DPM / GT coverage for each track #
            ########################################


            ################## PFADE ANPASSEN #######################

            gt_file = osp.join("data/MOT17Labels", "train", s, "gt", "gt.txt")
            # det_file = osp.join("../data/MOT17Labels", "train", s, "det", "det.txt")
            res_file = osp.join("output/tracker/MOT17", t, s+".txt")

            # gt_file = osp.join("MOT17Labels", "train", s, "gt", "gt.txt")
            # #det_file = osp.join(cfg.DATA_DIR, "MOT17Labels", "train", s, "det", "det.txt")
            # res_file = osp.join("track_missed_found", t, s+".txt")

            #########################################################

            #gtDB = read_txt_to_struct(gt_file)
            #gtDB = gtDB[gtDB[:,7] == 1]

            stDB = read_txt_to_struct(res_file)
            gtDB = read_txt_to_struct(gt_file)
            #dtDB = read_txt_to_struct(det_file)

            gtDB, distractor_ids = extract_valid_gt_data(gtDB)

            _, M, gtDB = evaluate_new(stDB, gtDB, distractor_ids)

            # reaload gtDB as entries not found are deleted
            gtDB_unf = read_txt_to_struct(gt_file)
            # filter out so that confidence and id = 1
            gtDB_unf = gtDB_unf[gtDB_unf[:,7] == 1]
            gtDB_unf = gtDB_unf[gtDB_unf[:,6] == 1]

            if t == tracker[0]:
                for vis in gtDB_unf[:,8]:
                    gt_vis.append(vis)
                gt_temp = gtDB_unf[gtDB_unf[:,8]>=0.9]
                heights = gt_temp[:,5] - gt_temp[:,3]
                if "-05-" in s:
                    heights *= 2.25
                for h in heights:
                    gt_h.append(h)

            gt_frames = np.unique(gtDB[:, 0])
            st_ids = np.unique(stDB[:, 1])
            gt_ids = np.unique(gtDB[:, 1])
            #dt_ids = np.unique(dtDB[:, 1])
            f_gt = len(gt_frames)
            n_gt = len(gt_ids)
            n_st = len(st_ids)

            #gt_inds = [{} for i in range(f_gt)]
            #st_inds = [{} for i in range(f_gt)]
            #dt_inds = [{} for i in range(f_gt)]
            gt_inds_unf = [{} for i in range(f_gt)]

            #D = [{} for i in range(f_gt)] #map detections to gt

            # hash the indices to speed up indexing
            #for i in range(gtDB.shape[0]):
            #    frame = np.where(gt_frames == gtDB[i, 0])[0][0]
            #    gid = np.where(gt_ids == gtDB[i, 1])[0][0]
            #    gt_inds[frame][gid] = i

            for i in range(gtDB_unf.shape[0]):
                frame = np.where(gt_frames == gtDB_unf[i, 0])[0][0]
                gt_id = gtDB_unf[i,1]
                gt_inds_unf[frame][gt_id] = i

            #gt_frames_list = list(gt_frames)
            #for i in range(stDB.shape[0]):
                # sometimes detection missed in certain frames, thus should be assigned to groundtruth frame id for alignment
            #    frame = gt_frames_list.index(stDB[i, 0])
            #    sid = np.where(st_ids == stDB[i, 1])[0][0]
            #    st_inds[frame][sid] = i

            for frame, m in enumerate(M):
                for gt_id, line in gt_inds_unf[frame].items():

                    vis = gtDB_unf[line, 8]
                    height = gtDB_unf[line, 5] - gtDB_unf[line, 3]
                    if "-05-" in s:
                        height *= 2.25

                    gid = -1
                    gid_ind = np.where(gt_ids == gt_id)[0]
                    if len(gid_ind) > 0:
                        gid = gid_ind[0]

                    # check if tracked
                    if gid in m.keys():
                        tracked_visibilities.append(vis)
                        tracked_visibilities_heights.append(height)
                        continue
                    else:
                        missed_visibilities.append(vis)
                        missed_visibilities_heights.append(height)

                    # get distance to last time tracked
                    last_non_empty = -1
                    for j in range(frame, -1, -1):
                        if gid in M[j].keys():
                            last_non_empty = j
                            break
                    next_non_empty = -1
                    for j in range(frame, f_gt):
                        if gid in M[j].keys():
                            next_non_empty = j
                            break

                    if next_non_empty != -1 and last_non_empty != -1:
                        gap_length = next_non_empty - last_non_empty - 1
                        missed_gaps_length.append(gap_length)


            """
            for i in range(dtDB.shape[0]):
                # sometimes detection missed in certain frames, thus should be assigned to groundtruth frame id for alignment
                frame = gt_frames_list.index(dtDB[i, 0])
                did = np.where(dt_ids == dtDB[i, 1])[0][0]
                dt_inds[frame][did] = i


            # find track <-> gt
            for frame in range(f_gt):
                gt_ids = list(gt_inds[frame].keys())
                gt_i = list(gt_inds[frame].values())
                gt_boxes = gtDB[gt_i, 2:6]

                st_ids = list(st_inds[frame].keys())
                st_i = list(st_inds[frame].values())
                st_boxes = stDB[st_i, 2:6]

                overlaps = np.zeros((len(st_i), len(gt_i)), dtype=float)
                for i in range(len(gt_i)):
                    overlaps[:, i] = bbox_overlap(st_boxes, gt_boxes[i])
                matched_indices = linear_assignment(1 - overlaps)

                for matched in matched_indices:
                    if overlaps[matched[0], matched[1]] > 0.5:
                        did = st_ids[matched[0]]
                        gid = gt_ids[matched[1]]
                        D[frame][did] = gid

            # Now check all detections if they were tracked
            for frame in range(f_gt):
                matched_dets = D[frame]

                for did, gid in matched_dets.items():
                    # not matched by tracker, get visbility
                    line = gt_inds[frame][gid]
                    vis = gtDB[line, 8]

                    # if matched only visibility is important
                    if gid in M[frame].keys():
                        tracked_visibilities.append(vis)
                        continue
                    else:
                        missed_visibilities.append(vis)

                    # get distance to last time tracked
                    last_non_empty = frame
                    for j in range(frame, -1, -1):
                        if gid in M[j].keys():
                            last_non_empty = j
                            break
                    next_non_empty = frame
                    for j in range(frame, f_gt):
                        if gid in M[j]:
                            next_non_empty = j
                            break

                    dist = min(frame-last_non_empty, next_non_empty-frame)
                    # check if not tracked at all ...
                    if dist > 0:
                        missed_distances.append(dist)"""

        missed_gaps_length = np.array(missed_gaps_length)
        missed_visibilities = np.array(missed_visibilities)
        tracked_visibilities = np.array(tracked_visibilities)
        tracked_visibilities_heights = np.array(tracked_visibilities_heights)
        missed_visibilities_heights = np.array(missed_visibilities_heights)

        #plt.plot([0,1], [0,1], 'r-')
        """
        plt.figure()
        plt.hist(tracked_visibilities, bins=20, density=False)
        plt.ylabel('occurence')
        plt.xlabel('visibility of target')
        plt.title('Boxes total: {}'.format(len(tracked_visibilities)))
        plt.savefig(osp.join(output_dir, "{}-{}-{}.pdf".format(t, detections, 'TR_VIS')), format='pdf')
        """

        #plt.plot([0,1], [0,1], 'r-')
        weights = np.zeros(missed_gaps_length.shape)
        for i in range(missed_gaps_length.shape[0]):
            weights[i] = 1/missed_gaps_length[i]
            #weights[i] = 1
        #plt.figure()
        #plt.xlim((0, 40))
        #bins = list(range(40))
        #plt.hist(missed_gaps_length, weights=weights, bins=bins, density=False)
        #plt.ylabel('occurence')
        #plt.xlabel('gap length in tracks')
        #plt.savefig(osp.join(output_dir, "{}-{}-{}.pdf".format('MISS_DIST', t, detections)), format='pdf')

        x = np.arange(1,41)
        y = np.zeros(len(x))
        ges_occurencies = 0
        for i,b in enumerate(x):
            occurencies = int((missed_gaps_length==b).sum())
            occurencies = occurencies/b
            ges_occurencies += occurencies
            y[i] = occurencies
        #y = y/ges_occurencies
        y_poly = np.poly1d(np.polyfit(x, y, 5))
        poly_missed_gaps_length.append(y_poly)

        """
        plt.figure()
        plt.hist(missed_visibilities, bins=20, density=False)
        plt.ylabel('occurence')
        #plt.xlim((0, xmax))
        plt.xlabel('visibility of target')
        plt.title('Boxes total: {}'.format(len(missed_visibilities)))
        plt.savefig(osp.join(output_dir, "{}-{}-{}.pdf".format(t, detections, 'MISS_VIS')), format='pdf')
        plt.close()
        """

        # combined plot
        # plt.figure()
        # plt.hist([tracked_visibilities, missed_visibilities], bins=20, density=True, label=['tracked', 'missed'])
        # plt.ylabel('occurence')
        # plt.xlabel('visibility of target')
        # plt.legend()
        #plt.title('Boxes total: {}'.format(len(tracked_visibilities)))
        # plt.savefig(osp.join(output_dir, "{}-{}-{}.pdf".format('VIS', t, detections)), format='pdf')

        # height 0.9 <= vis <= 1.0
        tracked_visibilities_heights = tracked_visibilities_heights[tracked_visibilities >= 0.9]
        missed_visibilities_heights = missed_visibilities_heights[missed_visibilities >= 0.9]
        # plt.figure()
        # plt.hist([tracked_visibilities_heights, missed_visibilities_heights], bins=15, density=True, label=['tracked', 'missed'])
        # plt.ylabel('occurence')
        # plt.xlabel('height of targets')
        # plt.legend()
        #plt.title('Boxes total: {}'.format(len(tracked_visibilities)))
        # plt.savefig(osp.join(output_dir, "{}-{}-{}.pdf".format('HEIGHTS', t, detections)), format='pdf')

        tv.append(tracked_visibilities)
        th.append(tracked_visibilities_heights)





    h_max = 250
    h_n_bins = 8
    v_n_bins = 12

    gt_h = np.array(gt_h)
    gt_vis = np.array(gt_vis)
    tv = np.array(tv)
    th = np.array(th)

    h_step = (h_max - np.min(gt_h)) / h_n_bins
    v_step = 1.0 / v_n_bins

    gt_h = gt_h[gt_h < h_max]

    h_bins = np.arange(0,h_n_bins) * h_step + np.min(gt_h)
    v_bins = np.arange(0,v_n_bins) * v_step

    gt_h_occurencies = np.zeros(h_bins.shape[0])
    for i in range(h_bins.shape[0]):
        lower = h_bins[i]
        tmp = gt_h[gt_h>=lower]
        if i != h_bins.shape[0]-1:
            upper = h_bins[i+1]
            tmp = tmp[tmp<upper]
        gt_h_occurencies[i] = len(tmp)

    gt_v_occurencies = np.zeros(v_bins.shape[0])
    for i in range(v_bins.shape[0]):
        lower = v_bins[i]
        tmp = gt_vis[gt_vis>=lower]
        if i != v_bins.shape[0]-1:
            upper = v_bins[i+1]
            tmp = tmp[tmp<upper]
        gt_v_occurencies[i] = len(tmp)

    heights_results = []
    for i in range(len(tracker)):
        t = tracker[i]
        heights = th[i]
        heights = heights[heights < h_max]
        res = []
        for i in range(h_bins.shape[0]):
            lower = h_bins[i]
            tmp = heights[heights >= lower]
            if i != h_bins.shape[0]-1:
                upper = h_bins[i+1]
                tmp = tmp[tmp < upper]
            percentage = len(tmp)/gt_h_occurencies[i]
            res.append(percentage)
        heights_results.append(np.array(res))


    vis_results = []
    for i in range(len(tracker)):
        t = tracker[i]
        vis = tv[i]
        res = []
        for i in range(v_bins.shape[0]):
            lower = v_bins[i]
            tmp = vis[vis >= lower]
            if i != v_bins.shape[0]-1:
                upper = v_bins[i+1]
                tmp = tmp[tmp < upper]
            percentage = len(tmp)/gt_v_occurencies[i]
            res.append(percentage)
        vis_results.append(np.array(res))


    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fontsize = 16
    tickfontsize = 12

    ############ AB HIER WICHTIG FÜR DICH ##########################
    # Plot heights
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax2.set_zorder(1)
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)
    ax2.spines['top'].set_visible(False)
    #sns.distplot(gt_h)
    #sns.kdeplot(gt_h, cut=0, color="red", shade=True, linewidth=0)

    sns.distplot(gt_h, bins=8,  color="red", norm_hist=False, ax=ax2, kde=False, hist_kws={"rwidth":0.95, 'range': (np.min(gt_h), h_max)})
    #plt.setp(ax3.get_xticklabels(), visible=False)
    #plt.setp(ax2.get_yticklabels(), visible=False)
    #ax2.tick_params(axis='y', which='both', length=0, labelsize=fontsize)
    ax2.tick_params(labelsize=tickfontsize)
    #ax2.set_ylabel('Occurrence in tracking ground truth (\%)')
    color='black'
    if "DPM" in detections:
        color='white'
    if "FRCNN" in detections:
        color='white'
    ax2.set_ylabel('Occurrences in ground truth', fontsize=fontsize, color=color)
    #ax2.set_ylim((0,h_max))
    #sns.kdeplot(gt_h, shade=True, cut=0)


    ax1.spines['top'].set_visible(False)
    bar_width = (0.7*h_step) / len(tracker)
    for i in range(len(tracker)):
        dis = (i - (len(tracker)-1)/2) * bar_width
        ax1.bar(h_bins + h_step/2 + dis, heights_results[i], bar_width, align='center', label=tracker[i].replace('_', '\_'))
    ax1.set_ylim((0, 1.0))
    ax1.set_xlim((np.min(gt_h),h_max))
    #if "DPM" in detections:
    #    plt.legend(fontsize=fontsize)
    plt.tick_params(labelsize=tickfontsize)

    color='black'
    if "DPM" in detections:
        color='white'
    if "SDP" in detections:
        color='white'
    ax1.set_xlabel('Object height (pixels)', fontsize=fontsize, color=color)
    color='black'
    if "FRCNN" in detections:
        color='white'
    if "SDP" in detections:
        color='white'
    ax1.set_ylabel('Tracked objects [\%]', fontsize=fontsize, color=color)
    #plt.title('height distribution from highly visible targets (>= 0.9)')

    plt.savefig(osp.join(output_dir, "heights09-{}.pdf".format(detections)), format='pdf', bbox_inches='tight')


    if "FRCNN" in detections:
        # Plot vis
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax2.set_zorder(1)
        ax1.set_zorder(2)
        ax1.patch.set_visible(False)
        ax2.spines['top'].set_visible(False)
        #sns.kdeplot(gt_vis, cut=0, color="red", shade=True, linewidth=0, ax=ax2)
        sns.distplot(gt_vis, bins=12,  color="red", norm_hist=False, ax=ax2, kde=False, hist_kws={"rwidth":0.95})
        ax2.tick_params(labelsize=tickfontsize)
        #plt.setp(ax2.get_yticklabels(), visible=False)
        #ax2.tick_params(axis='y', which='both', length=0)
        #ax2.set_ylabel('Occurrence in tracking ground truth (\%)')
        ax2.set_ylabel('Occurrences in ground truth', fontsize=fontsize)
        #ax2.ylim((0,0.006))

        ax1.spines['top'].set_visible(False)
        #plt.hist(vis_results, weights=vis_weights, label=tracker)
        #sns.distplot(gt_vis)
        #sns.kdeplot(gt_vis, bw=1.0, label="bw: 1.0", cut=0)
        #sns.kdeplot(gt_vis, bw=0.2, label="bw: 0.2", cut=0)
        #sns.kdeplot(gt_vis, bw=2, label="bw: 2", cut=0)
        bar_width = (0.7*v_step) / len(tracker)
        for i in range(len(tracker)):
            dis = (i - (len(tracker)-1)/2) * bar_width
            ax1.bar(v_bins + v_step/2 + dis, vis_results[i], bar_width, align='center', label=tracker[i].replace('_', '\_'))
        ax1.set_ylim((0,1.0))
        ax1.set_xlim((0,1.0))
        ax1.tick_params(labelsize=tickfontsize)
        ax1.legend(loc = 'upper left', fontsize=tickfontsize)
        ax1.set_xlabel('Object visibility [\%]', fontsize=fontsize)
        ax1.set_ylabel('Tracked objects [\%]', fontsize=fontsize)
        #plt.title('visibility distribution of all targets')

        plt.savefig(osp.join(output_dir, "vis-{}.pdf".format(detections)), format='pdf', bbox_inches='tight')

        """
        x_new = np.linspace(1, 40, num=101, endpoint=True)
        plt.figure()
        plt.ylabel('occurence')
        plt.xlabel('gap length in tracks')
        for y_poly, t in zip(poly_missed_gaps_length, tracker):
            plt.plot(x_new, y_poly(x_new), label=t)
        plt.legend()
        plt.savefig(osp.join(output_dir, "ALL-{}-{}.pdf".format(detections, 'MISS_DIST')), format='pdf')
        """
