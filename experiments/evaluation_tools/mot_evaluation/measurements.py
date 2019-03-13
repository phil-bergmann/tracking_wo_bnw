"""
2D MOT2016 Evaluation Toolkit
An python reimplementation of toolkit in 2DMOT16(https://motchallenge.net/data/MOT16/)

This file lists the matching algorithms.
1. clear_mot_hungarian: Compute CLEAR_MOT metrics

- Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics." Journal on Image and Video Processing 2008 (2008): 1.

2. idmeasures: Compute MTMC metrics

- Ristani, Ergys, et al. "Performance measures and a data set for multi-target, multi-camera tracking." European Conference on Computer Vision. Springer, Cham, 2016.



usage:
python evaluate_tracking.py 
    --bm                       Whether to evaluate multiple files(benchmarks)
    --seqmap [filename]        List of sequences to be evaluated
    --track  [dirname]         Tracking results directory: default path -- [dirname]/[seqname]/res.txt
    --gt     [dirname]         Groundtruth directory:      default path -- [dirname]/[seqname]/gt.txt
(C) Han Shen(thushenhan@gmail.com), 2018-02
"""

import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from .bbox import bbox_overlap
from easydict import EasyDict as edict
VERBOSE = False
def clear_mot_hungarian(stDB, gtDB, threshold):
    """
    compute CLEAR_MOT and other metrics
    [recall, precision, FAR, GT, MT, PT, ML, falsepositives, false negatives, idswitches, FRA, MOTA, MOTP, MOTAL]
    """
    st_frames = np.unique(stDB[:, 0])
    gt_frames = np.unique(gtDB[:, 0])
    st_ids = np.unique(stDB[:, 1])
    gt_ids = np.unique(gtDB[:, 1])
    #f_gt = int(max(max(st_frames), max(gt_frames)))  
    #n_gt = int(max(gt_ids)) 
    #n_st = int(max(st_ids)) 
    f_gt = len(gt_frames)
    n_gt = len(gt_ids)
    n_st = len(st_ids)

    mme = np.zeros((f_gt, ), dtype=float)          # ID switch in each frame
    c = np.zeros((f_gt, ), dtype=float)            # matches found in each frame
    fp = np.zeros((f_gt, ), dtype=float)           # false positives in each frame
    missed = np.zeros((f_gt, ), dtype=float)       # missed gts in each frame
    
    g = np.zeros((f_gt, ), dtype=float)            # gt count in each frame
    d = np.zeros((f_gt, n_gt), dtype=float)         # overlap matrix
    allfps = np.zeros((f_gt, n_st), dtype=float)
    
    gt_inds = [{} for i in range(f_gt)]
    st_inds = [{} for i in range(f_gt)]
    M = [{} for i in range(f_gt)]                  # matched pairs hashing gid to sid in each frame 
    
    switches = {i:{} for i in gt_frames}

    # hash the indices to speed up indexing
    for i in range(gtDB.shape[0]):
        frame = np.where(gt_frames == gtDB[i, 0])[0][0]
        gid = np.where(gt_ids == gtDB[i, 1])[0][0]
        gt_inds[frame][gid] = i

    gt_frames_list = list(gt_frames)
    for i in range(stDB.shape[0]):
        # sometimes detection missed in certain frames, thus should be assigned to groundtruth frame id for alignment
        frame = gt_frames_list.index(stDB[i, 0])
        sid = np.where(st_ids == stDB[i, 1])[0][0]
        st_inds[frame][sid] = i

    for t in range(f_gt):
        g[t] = len(gt_inds[t].keys()) 
        
        # preserving original mapping if box of this trajectory has large enough iou in avoid of ID switch
        if t > 0:
            mappings = list(M[t - 1].keys())
            sorted(mappings)
            for k in range(len(mappings)):
                if mappings[k] in gt_inds[t].keys() and M[t - 1][mappings[k]] in st_inds[t].keys():
                    row_gt = gt_inds[t][mappings[k]]
                    row_st = st_inds[t][M[t - 1][mappings[k]]]
                    dist = bbox_overlap(stDB[row_st, 2:6], gtDB[row_gt, 2:6])
                    if dist >= threshold:
                        M[t][mappings[k]] = M[t - 1][mappings[k]]
                        if VERBOSE:
                            print('perserving mapping: %d to %d'%(mappings[k], M[t][mappings[k]]))
        # mapping remaining groundtruth and estimated boxes
        unmapped_gt, unmapped_st  = [], []
        unmapped_gt = [key for key in gt_inds[t].keys() if key not in M[t].keys()]
        unmapped_st = [key for key in st_inds[t].keys() if key not in M[t].values()]
        if len(unmapped_gt) > 0 and len(unmapped_st) > 0: 
            overlaps = np.zeros((n_gt, n_st), dtype=float)
            for i in range(len(unmapped_gt)):
                row_gt = gt_inds[t][unmapped_gt[i]]
                for j in range(len(unmapped_st)):
                    row_st = st_inds[t][unmapped_st[j]]
                    dist = bbox_overlap(stDB[row_st, 2:6], gtDB[row_gt, 2:6])
                    if dist[0] >= threshold:
                        overlaps[i][j] = dist[0]
            matched_indices = linear_assignment(1 - overlaps)
            
            for matched in matched_indices:
                if overlaps[matched[0], matched[1]] == 0:
                    continue
                M[t][unmapped_gt[matched[0]]] = unmapped_st[matched[1]]
                if VERBOSE:
                    print('adding mapping: %d to %d'%(unmapped_gt[matched[0]], M[t][unmapped_gt[matched[0]]]))

        # compute statistics
        cur_tracked = list(M[t].keys())
        st_tracked = M[t].values()
        fps = [key for key in st_inds[t].keys() if key not in M[t].values()] 
        for k in range(len(fps)):
            allfps[t][fps[k]] = fps[k]
        # check miss match errors
        if t > 0:
            for i in range(len(cur_tracked)):
                ct = cur_tracked[i]
                est = M[t][ct]
                last_non_empty = -1
                for j in range(t - 1, 0, -1):
                    if ct in M[j].keys():
                        last_non_empty = j
                        break
                if ct in gt_inds[t - 1].keys() and last_non_empty != -1:
                    mtct, mlastnonemptyct = -1, -1
                    if ct in M[t]:
                        mtct = M[t][ct]
                    if ct in M[last_non_empty]:
                        mlastnonemptyct = M[last_non_empty][ct]
                    if mtct != mlastnonemptyct:
                        mme[t] += 1
                        #switches[t][mlastnonemptyct] = mtct

                        #row_st = st_inds[t][M[t - 1][mappings[k]]]
                        #stDB[row_st, 2:6]
                        row_new = st_inds[t][mtct]
                        row_old = st_inds[last_non_empty][mlastnonemptyct]
                        id_new = int(stDB[row_new, 1])
                        id_old = int(stDB[row_old, 1])
                        
                        # extract height and visibility at dissapearing point
                        gt_row = gt_inds[last_non_empty][ct]
                        gt_id = int(gtDB[gt_row, 1])
                        gt_height = float(gtDB[gt_row, 5]) - float(gtDB[gt_row, 3])
                        gt_vis = float(gtDB[gt_row, 8])
                        
                        switches[int(gt_frames[t])][id_new] = [id_old, int(gt_frames[last_non_empty]), gt_id, gt_height, gt_vis]

        c[t] = len(cur_tracked)
        fp[t] = len(st_inds[t].keys())
        fp[t] -= c[t]
        missed[t] = g[t] - c[t]
        for i in range(len(cur_tracked)):
            ct = cur_tracked[i]
            est = M[t][ct]
            row_gt = gt_inds[t][ct]
            row_st = st_inds[t][est]
            d[t][ct] = bbox_overlap(stDB[row_st, 2:6], gtDB[row_gt, 2:6])

    # calculate mean vis and mean height over all matches
    gt_height_list = []
    gt_vis_list = []
    for t,v in enumerate(M):
        for gt in v.keys():
            gt_row = gt_inds[t][gt]
            gt_height_list.append(float(gtDB[gt_row, 5]) - float(gtDB[gt_row, 3]))
            gt_vis_list.append(float(gtDB[gt_row, 8]))

    gt_tracked = {}
    for t,v in enumerate(M):
        for gt in v.keys():
            if gt not in gt_tracked:
                gt_tracked[gt] = []
            gt_tracked[gt].append(t)
    missed_height = []
    missed_vis = []
    missed_dist = []
    for gt, times in gt_tracked.items():
        times = np.array(times)
        t0 = times[0]
        t1 = times[-1]
        for t in range(t0+1,t1):
            if t in times:
                continue
            ind = np.where((times<=t)==True)[0][-1]
            missed_dist.append(np.min([t-times[ind], times[ind+1]-t]))
            gt_row = gt_inds[t][gt]
            height = float(gtDB[gt_row, 5]) - float(gtDB[gt_row, 3])
            vis = float(gtDB[gt_row, 8])
            missed_height.append(height)
            missed_vis.append(vis)


    return mme, c, fp, g, missed, d, M, allfps, \
            (switches, np.mean(gt_height_list), np.mean(gt_vis_list), np.mean(missed_height), np.mean(missed_vis), np.mean(missed_dist))

def idmeasures(gtDB, stDB, threshold):
    """
    compute MTMC metrics
    [IDP, IDR, IDF1]
    """
    st_ids = np.unique(stDB[:, 1])
    gt_ids = np.unique(gtDB[:, 1])
    n_st = len(st_ids)
    n_gt = len(gt_ids)
    groundtruth = [gtDB[np.where(gtDB[:, 1] == gt_ids[i])[0], :] for i in range(n_gt)]
    prediction = [stDB[np.where(stDB[:, 1] == st_ids[i])[0], :] for i in range(n_st)]
    cost = np.zeros((n_gt + n_st, n_st + n_gt), dtype=float)
    cost[n_gt:, :n_st] = float('inf')
    cost[:n_gt, n_st:] = float('inf')
    
    fp = np.zeros(cost.shape)
    fn = np.zeros(cost.shape)
    # cost matrix of all trajectory pairs
    cost_block, fp_block, fn_block = cost_between_gt_pred(groundtruth, prediction, threshold)

    cost[:n_gt, :n_st] = cost_block
    fp[:n_gt, :n_st] = fp_block
    fn[:n_gt, :n_st] = fn_block

    # computed trajectory match no groundtruth trajectory, FP
    for i in range(n_st):
        cost[i + n_gt, i] = prediction[i].shape[0]
        fp[i + n_gt, i] = prediction[i].shape[0]
    
    # groundtruth trajectory match no computed trajectory, FN
    for i in range(n_gt):
        cost[i, i + n_st] = groundtruth[i].shape[0]
        fn[i, i + n_st] = groundtruth[i].shape[0]
    
    
    matched_indices = linear_assignment(cost)

    nbox_gt = sum([groundtruth[i].shape[0] for i in range(n_gt)])
    nbox_st = sum([prediction[i].shape[0] for i in range(n_st)])
    
    IDFP = 0
    IDFN = 0
    for matched in matched_indices:
        IDFP += fp[matched[0], matched[1]]
        IDFN += fn[matched[0], matched[1]]
    IDTP = nbox_gt - IDFN
    assert IDTP == nbox_st - IDFP
    IDP = IDTP / (IDTP + IDFP) * 100               # IDP = IDTP / (IDTP + IDFP)
    IDR = IDTP / (IDTP + IDFN) * 100               # IDR = IDTP / (IDTP + IDFN)
    IDF1 = 2 * IDTP / (nbox_gt + nbox_st) * 100    # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

    measures = edict()
    measures.IDP = IDP
    measures.IDR = IDR
    measures.IDF1 = IDF1
    measures.IDTP = IDTP
    measures.IDFP = IDFP
    measures.IDFN = IDFN
    measures.nbox_gt = nbox_gt
    measures.nbox_st = nbox_st

    return measures

def corresponding_frame(traj1, len1, traj2, len2):
    """
    Find the matching position in traj2 regarding to traj1
    Assume both trajectories in ascending frame ID
    """
    p1, p2 = 0, 0
    loc = -1 * np.ones((len1, ), dtype=int)
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

def compute_distance(traj1, traj2, matched_pos):
    """
    Compute the loss hit in traj2 regarding to traj1
    """
    distance = np.zeros((len(matched_pos), ), dtype=float)
    for i in range(len(matched_pos)):
        if matched_pos[i] == -1:
            continue
        else:
            iou = bbox_overlap(traj1[i, 2:6], traj2[matched_pos[i], 2:6])
            distance[i] = iou
    return distance

def cost_between_trajectories(traj1, traj2, threshold):
    [npoints1, dim1] = traj1.shape
    [npoints2, dim2] = traj2.shape
    # find start and end frame of each trajectories
    start1 = traj1[0, 0]
    end1 = traj1[-1, 0]
    start2 = traj2[0, 0]
    end2 = traj2[-1, 0]

    ## check frame overlap
    has_overlap = max(start1, start2) < min(end1, end2)
    if not has_overlap:
        fn = npoints1
        fp = npoints2
        return fp, fn
    
    # gt trajectory mapping to st, check gt missed
    matched_pos1 = corresponding_frame(traj1[:, 0], npoints1, traj2[:, 0], npoints2)
    # st trajectory mapping to gt, check computed one false alarms
    matched_pos2 = corresponding_frame(traj2[:, 0], npoints2, traj1[:, 0], npoints1)
    dist1 = compute_distance(traj1, traj2, matched_pos1)
    dist2 = compute_distance(traj2, traj1, matched_pos2)
    # FN
    fn = sum([1 for i in range(npoints1) if dist1[i] < threshold]) 
    # FP
    fp = sum([1 for i in range(npoints2) if dist2[i] < threshold]) 
    return fp, fn

def cost_between_gt_pred(groundtruth, prediction, threshold):
    n_gt =  len(groundtruth)
    n_st = len(prediction)
    cost = np.zeros((n_gt, n_st), dtype=float)
    fp = np.zeros((n_gt, n_st), dtype=float)
    fn = np.zeros((n_gt, n_st), dtype=float)
    for i in range(n_gt):
        for j in range(n_st):
            fp[i, j], fn[i, j] = cost_between_trajectories(groundtruth[i], prediction[j], threshold)
            cost[i, j] = fp[i, j] + fn[i, j]
    return cost, fp, fn
