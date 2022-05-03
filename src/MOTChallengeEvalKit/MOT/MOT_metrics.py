import math
from collections import defaultdict
from Metrics import Metrics





class MOTMetrics(Metrics):
	def __init__(self, seqName = None):
		super().__init__()
		if seqName:
			self.seqName = seqName
		else: self.seqName = 0
		# Evaluation metrics

		self.register(name = "MOTA", formatter='{:.2f}'.format, )
		self.register(name = "MOTP", formatter='{:.2f}'.format)
		self.register(name = "MOTAL", formatter='{:.2f}'.format,write_mail = False)

		self.register(name = "IDF1", formatter='{:.2f}'.format)
		self.register(name = "IDP", formatter='{:.2f}'.format)
		self.register(name = "IDR", formatter='{:.2f}'.format)
		self.register(name = "IDTP", formatter='{:.0f}'.format, write_mail = False)
		self.register(name = "IDFP", formatter='{:.0f}'.format, write_mail = False)
		self.register(name = "IDFN", formatter='{:.0f}'.format, write_mail = False)



		self.register(name = "recall", display_name="Rcll", formatter='{:.2f}'.format)
		self.register(name = "precision", display_name="Prcn", formatter='{:.2f}'.format)



		self.register(name = "tp", display_name="TP", formatter='{:.0f}'.format)  # number of true positives
		self.register(name = "fp", display_name="FP", formatter='{:.0f}'.format) # number of false positives
		self.register(name = "fn", display_name="FN", formatter='{:.0f}'.format)  # number of false negatives

		self.register(name = "MTR", formatter='{:.2f}'.format)
		self.register(name = "PTR", formatter='{:.2f}'.format)
		self.register(name = "MLR", formatter='{:.2f}'.format)


		self.register(name = "MT", formatter='{:.0f}'.format)
		self.register(name = "PT", formatter='{:.0f}'.format)
		self.register(name = "ML", formatter='{:.0f}'.format)

		self.register(name = "F1", display_name="F1", formatter='{:.2f}'.format, write_mail = False)
		self.register(name = "FAR", formatter='{:.2f}'.format)
		self.register(name = "total_cost", display_name="COST", formatter='{:.0f}'.format, write_mail = False)
		self.register(name = "FM", formatter='{:.0f}'.format)
		self.register(name = "fragments_rel", display_name="FMR", formatter='{:.2f}'.format)

		self.register(name = "id_switches", display_name="IDSW", formatter='{:.0f}'.format)
		self.register(name = "id_switches_rel", display_name="IDSWR", formatter='{:.1f}'.format)

		self.register(name = "n_gt_trajectories", display_name = "GT",formatter='{:.0f}'.format,  write_mail = False)
		self.register(name = "n_tr_trajectories", display_name = "TR", formatter='{:.0f}'.format, write_db = False, write_mail = False)
		self.register(name = "total_num_frames",  display_name = "TOTAL_NUM", formatter='{:.0f}'.format, write_mail = False, write_db = False)


		self.register(name = "n_gt", display_name = "GT_OBJ", formatter='{:.0f}'.format, write_mail = False, write_db = False) # number of ground truth detections
		self.register(name = "n_tr", display_name = "TR_OBJ", formatter='{:.0f}'.format, write_mail = False, write_db = False) # number of tracker detections minus ignored tracker detections




	def compute_clearmot(self):
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
	        self.MOTA = -float("inf")


	    else:
	        self.MOTA = (1 - (self.fn + self.fp + self.id_switches) / float(self.n_gt) ) * 100.

	    if self.tp == 0:
	        self.MOTP = 0
	    else:
	        self.MOTP = (1-  self.total_cost / float(self.tp)) * 100.
	    if self.n_gt != 0:
	        if self.id_switches == 0:
	            self.MOTAL = (1 - (self.fn + self.fp + self.id_switches)  / float(self.n_gt) ) * 100.0
	        else:
	            self.MOTAL = (1 - (self.fn + self.fp + math.log10(self.id_switches)) / float(self.n_gt) ) * 100.

	    # calculate relative IDSW and FM
	    if self.recall != 0:
	        self.id_switches_rel = self.id_switches/self.recall
	        self.fragments_rel = self.FM/self.recall

	    else:
	        self.id_switches_rel = 0
	        self.fragments_rel = 0

		# ID measures

	    IDPrecision = self.IDTP / (self.IDTP + self.IDFP)
	    IDRecall = self.IDTP / (self.IDTP + self.IDFN)
	    self.IDF1 = 2*self.IDTP/(self.n_gt + self.n_tr)
	    if self.n_tr==0: IDPrecision = 0
	    self.IDP = IDPrecision * 100
	    self.IDR = IDRecall * 100
	    self.IDF1 = self.IDF1 * 100


	    if self.n_gt_trajectories == 0:
	        self.MTR = 0.
	        self.PTR = 0.
	        self.MLR = 0.
	    else:
	        self.MTR = self.MT * 100. / float(self.n_gt_trajectories)
	        self.PTR = self.PT * 100. / float(self.n_gt_trajectories)
	        self.MLR = self.ML * 100. / float(self.n_gt_trajectories)

	def compute_metrics_per_sequence(self, sequence, pred_file, gt_file, gtDataDir, benchmark_name):
		import matlab.engine
		try:
			eng = matlab.engine.start_matlab()
			print("MATLAB successfully connected")
		except:
			raise Exception("<br> MATLAB could not connect! <br>")

		eng.addpath("matlab_devkit/",nargout=0)

		results = eng.evaluateTracking(sequence, pred_file, gt_file, gtDataDir, benchmark_name , nargout = 5)
		eng.quit()
		update_dict = results[4]
		self.update_values(update_dict)