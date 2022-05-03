import sys, os
sys.path.append(os.path.abspath(os.getcwd()))
import math
from collections import defaultdict
from MOT_metrics import MOTMetrics
from Evaluator import Evaluator, run_metrics
import multiprocessing as mp
import pandas as pd


class MOT_evaluator(Evaluator):
	def __init__(self):
		self.type = "MOT"
	def eval(self):

		print("Check prediction files")
		error_message = ""
		for pred_file in self.tsfiles:
			print(pred_file)
			# check if file is comma separated
			df = pd.read_csv(pred_file, header = None, sep = ",")
			if len(df.columns) == 1:
				f = open(pred_file, "r")
				error_message+= "<exc>Submission %s not in correct form. Values in file must be comma separated.<br>Current form:<br>%s<br>%s<br>.........<br><!exc>" % (pred_file.split("/")[-1], f.readline(),  f.readline())
				raise Exception(error_message)

			df.groupby([0,1]).size().head()
			count = df.groupby([0,1]).size().reset_index(name='count')

			# check if any duplicate IDs
			if any( count["count"]>1):
			    doubleIDs  = count.loc[count["count"]>1][[0,1]].values
			    error_message+= "<exc> Found duplicate ID/Frame pairs in sequence %s." % pred_file.split("/")[-1]
			    for id in doubleIDs:
			        double_values = df[((df[0]==id[0] )&( df[1]==id[1]))]
			        for row in double_values.values:
			            error_message+="<br> %s" % row

			    error_message+="<br> <!exc> "
		if error_message != "":
			raise Exception(error_message)
		print("Files are ok!")

		arguments = []


		for seq, res, gt in zip(self.sequences, self.tsfiles, self.gtfiles):

			arguments.append({"metricObject": MOTMetrics(seq), "args" : {
			"gtDataDir":  os.path.join( self.datadir,seq),
			"sequence": str(seq) ,
			"pred_file":res,
			"gt_file": gt,
			"benchmark_name": self.benchmark_name}})
		try:
			if self.MULTIPROCESSING:
				p = mp.Pool(self.NR_CORES)
				print("Evaluating on {} cpu cores".format(self.NR_CORES))
				processes = [p.apply_async(run_metrics, kwds=inp) for inp in arguments]
				self.results = [p.get() for p in processes]
				p.close()
				p.join()

			else:
				self.results = [run_metrics(**inp) for inp in arguments]
			self.failed = False
		except:
			self.failed = True
			raise Exception("<exc> MATLAB evalutation failed <!exc>")
		self.Overall_Results = MOTMetrics("OVERALL")


if __name__ == "__main__":
	eval = MOT_evaluator()

	benchmark_name = "MOT16"
	gt_dir = "data/MOT16"
	res_dir = "res/MOT16res"
	eval_mode = "train"
	eval.run(
	    benchmark_name = benchmark_name,
	    gt_dir = gt_dir,
	    res_dir = res_dir,
	    eval_mode = eval_mode)
