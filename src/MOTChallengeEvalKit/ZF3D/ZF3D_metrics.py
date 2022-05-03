"""
Author: Malte Pedersen and Joakim Bruslund Haurum
E-mail author: mape@create.aau.dk
Last Modified: May 7 2020
"""

import math
import pandas as pd
import motmetrics as mm
import numpy as np
from collections import defaultdict
from Metrics import Metrics


"""
Create individual metrics class for new challenge. The new metrics class inherits functionality from the parent class Metrics
"""

class Zef_3dMetrics(Metrics):
    def __init__(self, seqName = None):
        super().__init__()
        if seqName:
            self.seqName = seqName
        else: self.seqName = 0

        # The maximum allowed distance in centimetres between detections and
        # ground truth positions
        self.thresh_3d = 0.5

        print('Registrering metrics for sequence {0}'.format(self.seqName))


        """
        Register metrics for the evaluation script
        E.g.
            self.register(name = "MOTA", formatter='{:.2f}'.format )
            self.register(name = "recall", display_name="Rcll", formatter='{:.2f}'.format)
        """
        self.register(name = "mota", formatter = '{:.2f}'.format, display_name = 'MOTA')
        self.register(name = "motp", formatter = '{:.2f}'.format, display_name = 'MOTP')
        self.register(name = "motal", formatter = '{:.2f}'.format, display_name = 'MOTAL', write_mail = False)

        self.register(name = "recall", formatter = '{:.2f}'.format, display_name = 'Rcll')
        self.register(name = "precision", formatter = '{:.2f}'.format, display_name = 'Prcn')
        self.register(name = "f1", formatter = '{:.2f}'.format, display_name = 'F1')
        self.register(name = "FAR", formatter='{:.2f}'.format)
        self.register(name = "fp", formatter = '{:d}'.format, display_name = 'FP')
        self.register(name = "tp", formatter = '{:d}'.format, display_name = 'TP')
        self.register(name = "fn", formatter = '{:d}'.format, display_name = 'FN')

        self.register(name = "n_gt_trajectories", display_name = "GT",formatter='{:.0f}'.format)
        self.register(name = "n_gt", display_name = "GT_OBJ", formatter='{:.0f}'.format, write_mail = False, write_db = False) # number of ground truth detections
        self.register(name = "num_objects", formatter = '{:d}'.format,  display_name = "GT", write_db = True, write_mail = False) # tp+fn
        self.register(name = "num_switches", formatter = '{:d}'.format, display_name = "IDSW")
        self.register(name = "idsw_ratio", formatter = '{:.1f}'.format, display_name = "IDSWR")

        self.register(name = "total_num_frames",  display_name = "TOTAL_NUM", formatter='{:.0f}'.format, write_mail = False, write_db = False)

        self.register(name = "num_predictions", formatter = '{:.1f}'.format, write_db = False, write_mail = False)
        self.register(name = "dist", formatter = '{:.2f}'.format, write_db = False, write_mail = False)
        self.register(name = "frag", formatter = '{:d}'.format, display_name = "FM")
        self.register(name = "fragments_rel", display_name="FMR", formatter='{:.2f}'.format)

        self.register(name = "mtbf_s", formatter = '{:.2f}'.format, display_name = "MTBFs")
        self.register(name = "mtbf_m", formatter = '{:.2f}'.format, display_name = "MTBFm")
        self.register(name = "mtbf_ssum", formatter = '{:.2f}'.format, write_db=False, write_mail=False)
        self.register(name = "mtbf_slen", formatter = '{:.2f}'.format, write_db=False, write_mail=False)
        self.register(name = "mtbf_nslen", formatter = '{:.2f}'.format, write_db=False, write_mail=False)
        self.register(name = "idfp", formatter = '{:.1f}'.format, display_name = "IDFP", write_mail = False)
        self.register(name = "idfn", formatter = '{:.1f}'.format, display_name = "IDFN", write_mail = False)
        self.register(name = "idtp", formatter = '{:.1f}'.format, display_name = "IDTP")
        self.register(name = "idp", formatter = '{:.1f}'.format, display_name = "IDP")
        self.register(name = "idr", formatter = '{:.1f}'.format, display_name = "IDR")
        self.register(name = "idf1", formatter = '{:.1f}'.format, display_name = "IDF1")
        self.register(name = "mt", formatter = '{:d}'.format, display_name = "MT")
        self.register(name = "ml", formatter = '{:d}'.format, display_name = "ML")
        self.register(name = "pt", formatter = '{:d}'.format, display_name = "PT")


        self.register(name = "MTR", formatter='{:.2f}'.format)
        self.register(name = "PTR", formatter='{:.2f}'.format)
        self.register(name = "MLR", formatter='{:.2f}'.format)



    def compute_clearmot(self):
        """ Compute clear mot metric for the benchmark
            E.g.
            # precision/recall etc.
            if (self.fp + self.tp) == 0 or (self.tp + self.fn) == 0:
               self.recall = 0.
               self.precision = 0.
            else:
               self.recall = (self.tp / float(self.tp + self.fn) ) * 100.
               self.precision = (self.tp / float(self.fp + self.tp) ) * 100.
        """
        # precision/recall
        if (self.fp + self.tp) == 0 or self.num_objects == 0:
            self.recall = 0.
            self.precision = 0.
        else:
            self.recall = (self.tp / float(self.num_objects) ) * 100.
            self.precision = (self.tp / float(self.fp + self.tp) ) * 100.

        # F1-score
        if (self.precision + self.recall) == 0:
            self.f1 = 0.
        else:
            self.f1 = (2 * self.precision * self.recall) / (self.precision + self.recall)
        # False alarm rate
        if self.total_num_frames == 0:
            self.FAR = "n/a"
        else:
            self.FAR = ( self.fp / float(self.total_num_frames) )
        # ID switch ratio
        if self.recall == 0:
            self.idsw_ratio = 0.
            self.fragments_rel = 0
        else:
            self.idsw_ratio = self.num_switches / (self.recall / 100.)
            self.fragments_rel = self.frag/self.recall
        # MOTA/MOTAL
        if self.num_objects == 0:
            self.mota = 0.
        else:
            self.mota = (1. - (self.fn + self.num_switches +
                              self.fp) / self.num_objects) * 100.
            self.motal = (1 - (self.fn + self.fp +
                          np.log10(self.num_switches + 1)) / self.num_objects) * 100.

        # MOTP
        if self.tp == 0:
            self.motp = -1.
        else:
            self.motp = (self.thresh_3d - (self.dist/self.tp)) * 100

        # ID precission/recall
        if (self.idfp + self.idtp) == 0 or (self.idtp + self.idfn) == 0:
            self.idr = 0.
            self.idp = 0.
        else:
            self.idr = (self.idtp / float(self.idtp + self.idfn) ) * 100.
            self.idp = (self.idtp / float(self.idfp + self.idtp) ) * 100.

        # IDF1
        if (self.num_objects + self.num_predictions) == 0:
            self.idf1 = 0.
        else:
            self.idf1 = float(2 * self.idtp) / (self.num_objects + self.num_predictions) * 100.

        # MTBF standard and monotonic
        if self.mtbf_slen == 0:
            self.mtbf_s = 0
            self.mtbf_m = 0
        else:
            self.mtbf_s = self.mtbf_ssum/self.mtbf_slen
            self.mtbf_m = self.mtbf_ssum/(self.mtbf_slen+self.mtbf_nslen)

        if self.n_gt_trajectories == 0:
            self.MTR = 0.
            self.PTR = 0.
            self.MLR = 0.
        else:
            self.MTR = self.mt * 100. / float(self.n_gt_trajectories)
            self.PTR = self.pt * 100. / float(self.n_gt_trajectories)
            self.MLR = self.ml * 100. / float(self.n_gt_trajectories)

    def compute_metrics_per_sequence(self, sequence, det_df, gt_df, gtDataDir, benchmark_name, **kwargs):
        """

        """
        maxDist = self.thresh_3d
        posFunc = self.get3Dpos
        distFunc = self.pairwiseDistance
        gt_frame_col = "frame"
        det_frame_col = "frame"

        gt_df = gt_df.dropna(subset=["3d_x", "3d_y", "3d_z"])
        det_df = det_df[(det_df["3d_x"] > 0)  &  (det_df["3d_y"] > 0)  &  (det_df["3d_z"] > 0)]

        # Get unique occurring frames
        gt_frames = gt_df[gt_frame_col].unique()
        det_frames = det_df[det_frame_col].unique()

        gt_frames = [int(x) for x in gt_frames]
        det_frames = [int(x) for x in det_frames]
        print( "det num frames")

        frames = list(set(gt_frames+det_frames))

        print("[Seq {}]\nAmount of GT frames: {}\nAmount of det frames: {}\nSet of all frames: {}".format(sequence, len(gt_frames), len(det_frames), len(frames)))

        acc = mm.MOTAccumulator(auto_id=False)

        dist_sum = 0

        try:
            for frame in frames:

                # Get the df entries for this specific frame
                gts = gt_df[gt_df[gt_frame_col] == frame]
                dets = det_df[det_df[det_frame_col] == frame]

                gt_data = True
                det_data = True

                # Get ground truth positions, if any
                if len(gts) > 0:
                    gt_pos, gt_ids = posFunc(gts)
                    gt_ids = [x for x in gt_ids]
                else:
                    gt_ids = []
                    gt_data = False

                # Get detections, if any
                if len(dets) > 0:
                    det_pos, det_ids = posFunc(dets)
                    det_ids = [x for x in det_ids]
                else:
                    det_ids = []
                    det_data = False

                # Get the L2 distance between ground truth positions, and the detections
                if gt_data and det_data:
                    dist = distFunc(gt_pos, det_pos, maxDist=maxDist).tolist()
                else:
                    dist = []

                # Update accumulator
                acc.update(gt_ids,              # Ground truth objects in this frame
                        det_ids,                # Detector hypotheses in this frame
                        dist,                   # Distance between ground truths and observations
                        frame)

                dist_sum += self.nestedSum(dist)
        except:
            print("Add some more information for the exception here.") # FIX
            raise Exception("<exc> Evaluation failed <!exc>")

        metrics = self.calcMetrics(acc)
        # get number of gt tracks
        self.n_gt_trajectories = int(len(gt_df["id"].unique()))

        # True/False positives
        self.tp = int(metrics['num_detections'])
        self.fp = int(metrics['num_false_positives'])

        # Total number of unique object appearances over all frames (tp + fn)
        self.num_objects = int(metrics['num_objects'])

        # Total number of misses
        self.fn = int(metrics['num_misses'])

        # Total number of track switches
        self.num_switches = int(metrics['num_switches'])

        # Total L2 distance between gt and dets
        self.dist = dist_sum

        # Total amount of fragments
        self.frag = int(metrics["num_fragmentations"])

        # MTBF sequences and null sequences
        self.mtbf_ssum = int(metrics["mtbf_ssum"])
        self.mtbf_slen = int(metrics["mtbf_slen"])
        self.mtbf_nslen = int(metrics["mtbf_nslen"])

        # ID true positives/false negatives/false positives
        self.idtp = int(metrics["idtp"])
        self.idfn = int(metrics["idfn"])
        self.idfp = int(metrics["idfp"])

        # Total number of unique prediction appearances
        self.num_predictions = int(metrics["num_predictions"])

        # Mostly tracked
        self.mt = int(metrics["mostly_tracked"])

        # Mostly lost
        self.ml = int(metrics["mostly_lost"])

        # Partially tracked
        self.pt = int(metrics["partially_tracked"])

        # total number of frames
        self.total_num_frames = int(metrics["num_frames"])

    def nestedSum(self, x):
        """
        Returns the summed elements in nested lists
        """
        total = 0
        for i in x:
            if isinstance(i, list):
                total += self.nestedSum(i)
            elif not math.isnan(i):
                total += i
            else:
                pass
        return total

    def get3Dpos(self, df):
        """
        Returns the 3D position in a dataset

        Input:
            df: Pandas dataframe

        Output:
            pos: Numpy array of size [n_ids, 3] containing the 3d position
            ids: List of IDs
        """
        ids = df["id"].unique()
        ids = [int(x) for x in ids]

        pos = np.zeros((len(ids), 3))
        for idx, identity in enumerate(ids):
            df_id = df[df["id"] == identity]
            pos[idx,0] = df_id["3d_x"]
            pos[idx,1] = df_id["3d_y"]
            pos[idx,2] = df_id["3d_z"]

        return pos, ids

    def pairwiseDistance(self, X,Y, maxDist):
        """
        X and Y are n x d and m x d matrices, where n and m are the amount of observations, and d is the dimensionality of the observations
        """

        X_ele, X_dim = X.shape
        Y_ele, Y_dim = Y.shape

        assert X_dim == Y_dim, "The two provided matrices not have observations of the same dimensionality"

        mat = np.zeros((X_ele, Y_ele))

        for row, posX in enumerate(X):
            for col, posY in enumerate(Y):
                mat[row, col] = np.linalg.norm(posX-posY)

        mat[mat > maxDist] = np.nan

        return mat

    def calcMetrics(self, acc):
        """
        Calculates all relevant metrics for the dataset

        Input:
            acc: MOT Accumulator object

        Output:
            summary: Pandas dataframe containing all the metrics
        """
        mh = mm.metrics.create()
        summary = mh.compute_many([acc],
                                  metrics=mm.metrics.motchallenge_metrics
                                  +["num_objects"]
                                  +["num_predictions"]
                                  +["num_frames"]
                                  +["num_detections"]
                                  +["num_fragmentations"]
                                  +["idfp"]
                                  +["idfn"]
                                  +["idtp"])

        summary["motal"] = self.MOTAL(summary)
        mtbf_ssum, mtbf_slen, mtbf_nslen = self.MTBF(acc.mot_events)
        summary["mtbf_ssum"] = mtbf_ssum # Sum of sequences
        summary["mtbf_slen"] = mtbf_slen # Number of sequences
        summary["mtbf_nslen"] = mtbf_nslen # Number of null sequences

        return summary

    def MOTAL(self, metrics):
        """
        Calculates the MOTA variation where the amount of id switches is
        attenuated by using the log10 function
        """
        return 1 - (metrics["num_misses"] + metrics["num_false_positives"] + np.log10(metrics["num_switches"]+1)) / metrics["num_objects"]


    def MTBF(self, events):
        """
        Calculates the Mean Time Between Failures (MTBF) metric from the motmetric events dataframe

        Input:
            events: Pandas Dataframe structured as per the motmetrics package

        Output:
            MTBF_s: The Standard MTBF metric proposed in the original paper
            MTBF_m: The monotonic MTBF metric proposed in the original paper
        """

        unique_gt_ids = events.OId.unique()
        seqs = []
        null_seqs = []
        for gt_id in unique_gt_ids:
            gt_events = events[events.OId == gt_id]

            counter = 0
            null_counter = 0

            for _, row in gt_events.iterrows():
                if row["Type"] == "MATCH":
                    counter += 1
                elif row["Type"] == "SWITCH":
                    seqs.append(counter)
                    counter = 1
                else:
                    seqs.append(counter)
                    counter = 0
                    null_counter = 1

                if counter > 0:
                    if null_counter > 0:
                        null_seqs.append(null_counter)
                        null_counter = 0

            if counter > 0:
                seqs.append(counter)
            if null_counter > 0:
                null_seqs.append(null_counter)

        seqs = np.asarray(seqs)
        seqs = seqs[seqs>0]

        mtbf_ssum = sum(seqs)
        mtbf_slen = len(seqs)
        mtbf_nslen = len(null_seqs)

        if mtbf_ssum == 0:
            return 0, 0, 0
        else:
            return mtbf_ssum, mtbf_slen, mtbf_nslen
