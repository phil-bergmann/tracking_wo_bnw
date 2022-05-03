import os
import math
from collections import defaultdict
from Metrics import Metrics
import numpy as np


class DETMetrics(Metrics):
	def __init__(self, seqName = None):
		super().__init__()
		if seqName:
			self.seqName = seqName
		else: self.seqName = 0
		# Evaluation metrics

		self.register(name = "MODA", formatter='{:.2f}'.format)
		self.register(name = "MODP", formatter='{:.2f}'.format)
		self.register(name = "AP", formatter='{:.2f}'.format)

		self.register(name = "recall", display_name="Rcll", formatter='{:.2f}'.format)
		self.register(name = "precision", display_name="Prcn", formatter='{:.2f}'.format)

		self.register(name = "tp", display_name="TP", formatter='{:.0f}'.format)  # number of true positives
		self.register(name = "fp", display_name="FP", formatter='{:.0f}'.format) # number of false positives
		self.register(name = "fn", display_name="FN", formatter='{:.0f}'.format)  # number of false negatives

		self.register(name = "F1", display_name="F1", formatter='{:.2f}'.format)
		self.register(name = "FAR", formatter='{:.2f}'.format)

		self.register(name = "n_gt_trajectories", display_name = "GT",formatter='{:.0f}'.format)
		#self.register(name = "n_tr_trajectories", display_name = "TR", formatter='{:.0f}'.format)
		self.register(name = "total_num_frames", display_name="TOTAL_NUM", formatter='{:.0f}'.format, write_db = False, write_mail = False)


		self.register(name = "n_gt", display_name = "GT_OBJ", formatter='{:.0f}'.format, write_db = False, write_mail = False) # number of ground truth detections
		#self.register(name = "n_tr", display_name = "TR_OBJ", formatter='{:.0f}'.format) # number of tracker detections minus ignored tracker detections

		self.cache(name= "td", func = np.hstack)
		self.cache(name= "ious", func = np.hstack)
		self.cache(name= "scores", func = np.hstack)
		self.cache(name= "tp_list", func = np.hstack)



	def compute_clearmot(self):
	    td = 0.5
	    # precision/recall etc.
	    if (self.fp + self.tp) == 0 or (self.tp + self.fn) == 0:
	        self.recall = 0.
	        self.precision = 0.
	    else:
	        self.recall = (self.tp / float(self.tp + self.fn) ) * 100.
	        self.precision = (self.tp / float(self.fp + self.tp) ) * 100.
	    if (self.recall + self.precision) == 0:
	        self.F1 = 0.
	    else:
	        self.F1 = 2. * (self.precision * self.recall) / (self.precision + self.recall)
	    if self.total_num_frames == 0:
	        self.FAR = "n/a"
	    else:
	        self.FAR = ( self.fp / float(self.total_num_frames) )
	    # compute CLEARMOT
	    if self.n_gt == 0:
	        self.MODA = -float("inf")
	    else:
	        self.MODA = (1 - (self.fn + self.fp ) / float(self.n_gt) ) * 100.

	    modp_sum = sum(self.ious[((self.ious>=td ) & ( self.ious<np.inf))])
	    if self.tp != 0 :
	        self.MODP=modp_sum/float(self.tp) * 100
	    else:
	        self.MODP = 0

	    order = np.argsort(self.scores)[::-1]
	    self.tp_list=self.tp_list[order]
	    FP = np.array(self.tp_list!=1.)
	    TP = np.cumsum(self.tp_list)
	    FP = np.cumsum(FP)


	    xs = TP/self.n_gt
	    ys = TP/(TP+FP)
	    xs1 = np.append(xs, np.inf)
	    ys1 = np.append(ys, 0)

	    self.RefPrcn = np.linspace( 0, 1, 11)
	    self.xs1 = xs1
	    self.ys1 = ys1

	    for i, r in enumerate(self.RefPrcn ):
	        j = np.where(xs1 >= r)[0]
	        self.RefPrcn[i] = ys1[j[0]]

	    self.AP = np.mean( self.RefPrcn )

	def compute_metrics_per_sequence(self, sequence, pred_file, gt_file, gtDataDir, benchmark_name):
		sum_tuple = lambda x: sum(map(sum_tuple, x)) if isinstance(x, tuple) else x
		tuple_to_list = lambda x: list(map(tuple_to_list, x)) if isinstance(x, tuple) else x
		import matlab.engine
		try:
			eng = matlab.engine.start_matlab()
			print("MATLAB successfully connected")
		except:
			raise Exception("<br> MATLAB could not connect! <br>")

		eng.addpath("src/MOTChallengeEvalKit/matlab_devkit/", nargout=0)
		# eng.addpath("matlab_devkit/", nargout=0)

		print("start matlab evaluation")


		results = eng.evaluateDetection(sequence, pred_file, gt_file, gtDataDir, benchmark_name, nargout=1)

		eng.quit()
		update_dict = results
		results["ious"] =  np.array( tuple_to_list(results["ious"] )).reshape(-1)
		results["scores"] = np.asarray(results['scores']).reshape(-1)
		results["tp_list"] = np.asarray(results['tp_list']).reshape(-1)
		self.update_values(update_dict)

