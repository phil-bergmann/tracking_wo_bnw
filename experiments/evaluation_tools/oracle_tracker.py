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
import csv

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from tracker.rfrcnn import FRCNN as rFRCNN
from tracker.vfrcnn import FRCNN as vFRCNN
from tracker.config import cfg, get_output_dir
from tracker.utils import plot_sequence
from tracker.mot_sequence import MOT_Sequence
from tracker.kitti_sequence import KITTI_Sequence
from tracker.datasets.factory import Datasets
from tracker.tracker_debug import Tracker
from tracker.utils import interpolate
from tracker.resnet import resnet50

from sklearn.utils.linear_assignment_ import linear_assignment
from easydict import EasyDict as edict
from mot_evaluation.io import read_txt_to_struct, read_seqmaps, extract_valid_gt_data, print_metrics
from mot_evaluation.bbox import bbox_overlap
from mot_evaluation.measurements import clear_mot_hungarian, idmeasures

ex = Experiment()

ex.add_config('experiments/cfgs/oracle_tracker.yaml')

# hacky workaround to load the corresponding cnn config and not having to hardcode it here
ex.add_config(ex.configurations[0]._conf['oracle_tracker']['siamese_config'])

Tracker = ex.capture(Tracker, prefix='oracle_tracker.tracker')

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
    print('[TRACK PREPROCESSING]: remove distractors and low visibility boxes, remaining %d/%d computed boxes'%(len(keep_idx), len(res_keep)))
    trackDB = trackDB[keep_idx, :]
    print('Distractors:', distractor_ids)
    #keep_idx = np.array([i for i in xrange(gtDB.shape[0]) if gtDB[i, 1] not in distractor_ids and gtDB[i, 8] >= minvis])
    keep_idx = np.array([i for i in range(gtDB.shape[0]) if gtDB[i, 6] != 0] )
    print('[GT PREPROCESSING]: Removing distractor boxes, remaining %d/%d computed boxes'%(len(keep_idx), gtDB.shape[0]))
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

    return metrics, extra_info, clear_mot_info, ML_PT_MT

   

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

def evaluate_new(results, gt_file):
    res = []
    for i, track in results.items():
        for frame, bb in track.items():
            x1 = bb[0]
            y1 = bb[1]
            x2 = bb[2]
            y2 = bb[3]
            res.append([float(frame+1), float(i+1), float(x1+1), float(y1+1), float(x2+1), float(y2+1), float(-1), float(-1), float(-1), float(-1)])

    trackDB = np.array(res)
    gtDB = read_txt_to_struct(gt_file)

    # manipulate mot15 to fit mot16
    #gtDB[:,7] = gtDB[:,6]
    #gtDB[:,6] = 1
    #gtDB[:,8] = 1
    #gtDB = gtDB[:,:9]

    gtDB, distractor_ids = extract_valid_gt_data(gtDB)

    metrics, extra_info, clear_mot_info, ML_PT_MT = evaluate_sequence(trackDB, gtDB, distractor_ids)

    print_metrics(' Evaluation', metrics)

    # evaluate sizes of ML_PT_MT
    ML_PT_MT_sizes = [[]]
    ML_PT_MT_vis = [[]]
    for i, p in enumerate(ML_PT_MT, 1):
        ML_PT_MT_sizes.append([])
        ML_PT_MT_vis.append([])
        for tr_id in p:
            gt_in_person = np.where(gtDB[:, 1] == tr_id)[0]
            #w = gtDB[gt_in_person, 4]
            h = gtDB[gt_in_person, 5] - gtDB[gt_in_person, 3]
            #sizes = (w*h).mean()
            ML_PT_MT_sizes[i].append(h.mean())
            ML_PT_MT_sizes[0].append(h.mean())

            vis = gtDB[gt_in_person, 8]
            ML_PT_MT_vis[i].append(vis.mean())
            ML_PT_MT_vis[0].append(vis.mean())

    print("\t\theight\tvis")
    print("MT: \t\t{:.2f}\t{:.2f}".format(np.mean(ML_PT_MT_sizes[3]), np.mean(ML_PT_MT_vis[3])))
    print("PT: \t\t{:.2f}\t{:.2f}".format(np.mean(ML_PT_MT_sizes[2]), np.mean(ML_PT_MT_vis[2])))
    print("ML: \t\t{:.2f}\t{:.2f}".format(np.mean(ML_PT_MT_sizes[1]), np.mean(ML_PT_MT_vis[1])))
    print("total (mean of\t{:.2f}\t{:.2f}\ntrack means):".format(np.mean(ML_PT_MT_sizes[0]), np.mean(ML_PT_MT_vis[0])))

    return clear_mot_info

@ex.automain
def my_main(oracle_tracker, siamese, _config):
    # set all seeds
    torch.manual_seed(oracle_tracker['seed'])
    torch.cuda.manual_seed(oracle_tracker['seed'])
    np.random.seed(oracle_tracker['seed'])
    torch.backends.cudnn.deterministic = True

    print(_config)

    ##########################
    # Initialize the modules #
    ##########################
    
    print("[*] Building FRCNN")

    if oracle_tracker['network'] == 'vgg16':
        frcnn = vFRCNN()
    elif oracle_tracker['network'] == 'res101':
        frcnn = rFRCNN(num_layers=101)
    else:
        raise NotImplementedError("Network not understood: {}".format(oracle_tracker['network']))

    frcnn.create_architecture(2, tag='default',
        anchor_scales=frcnn_cfg.ANCHOR_SCALES,
        anchor_ratios=frcnn_cfg.ANCHOR_RATIOS)
    frcnn.eval()
    frcnn.cuda()
    frcnn.load_state_dict(torch.load(oracle_tracker['frcnn_weights']))
    
    cnn = resnet50(pretrained=False, **siamese['cnn'])
    cnn.load_state_dict(torch.load(oracle_tracker['cnn_weights']))
    cnn.eval()
    cnn.cuda()
    tracker = Tracker(frcnn=frcnn, cnn=cnn)

    output_dir = osp.join(get_output_dir(oracle_tracker['module_name']), oracle_tracker['name'])

    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    print("[*] Beginning evaluation...")

    time_ges = 0

    #train = ["MOT17-13", "MOT17-11", "MOT17-10", "MOT17-09", "MOT17-05", "MOT17-04", "MOT17-02", ]

    analysis_results = []

    for db in Datasets(oracle_tracker['dataset']):
        tracker.reset()

        now = time.time()

        print("[*] Evaluating: {}".format(db))
        
        #db = MOT_Sequence(s)

        dl = DataLoader(db, batch_size=1, shuffle=False)
        for sample in dl:
            tracker.step(sample)
        results, debug = tracker.get_results()

        if oracle_tracker['interpolate']:
            results = interpolate(results)

        #print("[!] Killed {} tracks by NMS".format(tracker.nms_killed))

        print("Tracks found: {}".format(len(results)))
        print("[!] Killed {} tracks by NMS".format(tracker.nms_killed))
        #print("[*] Time needed for {} evaluation: {:.3f} s".format(s, time.time() - now))

        #db.write_results(results, osp.join(output_dir))
        
        #if oracle_tracker['write_images']:
        #    plot_sequence(results, db, osp.join(output_dir, s))
        gt_file = osp.join(cfg.DATA_DIR, "MOT17Labels", "train", str(db)+"-DPM", "gt", "gt.txt")
        (switches, gt_mean_height, gt_mean_vis, missed_height, missed_vis, missed_dist) = evaluate_new(results, gt_file)

        #with open(osp.join(output_dir, str(s)+"_switches.txt"), "w") as of:
        #    for t,sw in enumerate(switches):
        #        of.write("{}:       {}\n".format(t, sw))

        with open(osp.join(output_dir, str(db)+"_debug.txt"), "w") as of:
            for i, track in debug.items():
                of.write("Track id: {}\n".format(i))
                for im_index, data in track.items():
                    of.write("Frame: {}\n".format(im_index))
                    of.write("Pos: {}\n".format(data["pos"]))
                    of.write("{}".format(data["info"]))
                of.write("\n\n")


        sizes = []
        gt_ids = []
        gt_heights = []
        gt_viss = []
        reason_loose = { 'NMS':0, 'score':0, 'regression':0}
        reason_find = {'created':0, 'reid':0, 'regression':0}

        with open(osp.join(output_dir, str(db)+"_sum.txt"), "w") as of:
            #for f1,sw in enumerate(switches, 1):
            for f1, sw in switches.items():
                of.write("[*] Frame: {}\n".format(f1))
                for t1, (t0, f0, gt_id, gt_height, gt_vis) in sw.items():
                    gt_heights.append(gt_height)
                    gt_ids.append(gt_id)
                    gt_viss.append(gt_vis)
                    of.write("ID switch from track {} to {} with size {}\n".format(t0, t1, gt_height))
                    #of.write("Old Track: {}\n".format(debug[t0-1][f0-1]["pos"]))
                    #of.write("{}".format(debug[t0-1][f0-1]["info"]))
                    #debug[t1-1][f1-1]["pos"][3]-debug[t1-1][f1-1]["pos"][1]
                    of.write("Old Track:\n")
                    for frame, value in debug[t0-1].items():
                        if frame == f0-1 or frame == f0 or frame == f0+1:
                            of.write("{} {}\n{}".format(frame, value["pos"], value["info"]))
                        if frame == f0:
                            if "too low score" in value["info"]:
                                reason_loose['score'] += 1
                            elif "NMS" in value["info"]:
                                reason_loose['NMS'] += 1
                            elif "Regressing" in value["info"]:
                                reason_loose['regression'] += 1
                    #of.write("{} {}\n{}".format(f0, debug[t0-1][f0-1]["pos"], debug[t0-1][f0-1]["info"]))
                    #of.write("{} {}\n{}".format(f0, debug[t0-1][f0]["pos"], debug[t0-1][f0]["info"]))
                    #of.write("{} {}\n{}".format(f0, debug[t0][f0-1]["pos"], debug[t0][f0-1]["info"]))


                    of.write("\nNew Track:\n")
                    for frame, value in debug[t1-1].items():
                        if frame == f1-1 or frame == f1-2 or frame == f1-3:
                            of.write("{} {}\n{}".format(frame, value["pos"], value["info"]))
                        if frame == f1-1:
                            if "Created" in value['info']:
                                reason_find['created'] += 1
                            elif "ReIded" in value["info"]:
                                reason_find["reid"] += 1
                            elif "Regressing" in value["info"]:
                                reason_find['regression'] += 1
                    #of.write("{} {}\n{}".format(f1, debug[t1-1][f1-1]["pos"], debug[t1-1][f1-1]["info"]))

                    #of.write("New Track: {}\n".format(debug[t1-1][f1-1]["pos"]))
                    #of.write("{}\n".format(debug[t1-1][f1-1]["info"]))
                    of.write("\n")
                of.write("\n\n")
            #of.write("Sizes: {}\n".format(sizes))
        print("[*] Number of ID switches: {}".format(len(gt_ids)))
        print("[*] Number of gt targets involved in ID switches: {}".format(len(np.unique(gt_ids))))
        print("[*] Mean height of ID switches: {}".format(np.mean(gt_heights)))
        print("[*] Mean visibility of ID switches targets: {}".format(np.mean(gt_viss)))
        print("[*] Mean height of tracked boxes: {}".format(np.mean(gt_mean_height)))
        print("[*] Mean visibility of tracked boxes: {}".format(np.mean(gt_mean_vis)))
        print("[*] Reasons for ID swtiches:")
        print("\tLoose:")
        for k,v in reason_loose.items():
            print("\t\t{}: {}".format(k,v))
        print("\tFind:")
        for k,v in reason_find.items():
            print("\t\t{}: {}".format(k,v))
        print(gt_ids)


        detections = oracle_tracker['dataset'].split("_")[2]
        ar = [str(db)+"-"+detections, len(gt_ids), tracker.nms_killed, reason_loose['score'], reason_loose['NMS'],
               reason_loose['regression'], reason_find['created'], reason_find["reid"], reason_find['regression'],
               np.mean(gt_mean_height), np.mean(gt_mean_vis),
               np.mean(gt_heights), np.mean(gt_viss),
               missed_height, missed_vis, missed_dist]
            
        analysis_results.append(ar)

        print(ar)

        db.write_results(results, output_dir)

        if oracle_tracker['write_images']:
            plot_sequence(results, db, osp.join(output_dir, str(db)))
    
    # print results to csv
    file = osp.join(output_dir, oracle_tracker['dataset'])
    with open(file, "w") as of:
        writer = csv.writer(of, delimiter=' ')
        for row in analysis_results:
            row_new = list(map(lambda x: str(x).replace(".",","), row))
            writer.writerow(row_new)