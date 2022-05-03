import sys, os
sys.path.append(os.path.abspath(os.getcwd()))
import math
from collections import defaultdict
from MOTS_metrics import MOTSMetrics
from Evaluator import Evaluator, run_metrics

import multiprocessing as mp



class MOTS_evaluator(Evaluator):
	def __init__(self):
		self.type = "MOTS"
	def eval(self):

		arguments = []
		for seq, res, gt in zip(self.sequences, self.tsfiles, self.gtfiles):
			arguments.append({"metricObject": MOTSMetrics(seq),
			"args" : {"gtDataDir": os.path.join( self.datadir,seq),
			"sequence": str(seq) ,
			"pred_file":res,
			"gt_file": gt,
			"benchmark_name": self.benchmark_name}})

		if self.MULTIPROCESSING:
			p = mp.Pool(self.NR_CORES)
			processes = [p.apply_async(run_metrics, kwds=inp) for inp in arguments]
			self.results = [p.get() for p in processes]
			p.close()
			p.join()

		else:
			results = [run_metrics(**inp) for inp in arguments]


		# Sum up results for all sequences
		self.Overall_Results = MOTSMetrics("OVERALL")


if __name__ == "__main__":
	eval = MOTS_evaluator()
	benchmark_name = "MOTS"
	gt_dir = "data/MOTS"
	res_dir = "res/MOTSres"
	eval_mode = "train"
	eval.run(
	         benchmark_name = benchmark_name,
	         gt_dir = gt_dir,
	         res_dir = res_dir,
	         eval_mode = eval_mode)
