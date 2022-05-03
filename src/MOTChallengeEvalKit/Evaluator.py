import sys, os
sys.path.append(os.getcwd())
import argparse

import traceback
import time
import pickle
import pandas as pd
import glob
from os import path
import numpy as np



class Evaluator(object):
	""" The `Evaluator` class runs evaluation per sequence and computes the overall performance on the benchmark"""
	def __init__(self):
		pass

	def run(self, benchmark_name = None ,  gt_dir = None, res_dir = None, save_pkl = None, eval_mode = "train", seqmaps_dir = "seqmaps"):
		"""
		Params
		-----
		benchmark_name: Name of benchmark, e.g. MOT17
		gt_dir: directory of folders with gt data, including the c-files with sequences
		res_dir: directory with result files
			<seq1>.txt
			<seq2>.txt
			...
			<seq3>.txt
		eval_mode:
		seqmaps_dir:
		seq_file: File name of file containing sequences, e.g. 'c10-train.txt'
		save_pkl: path to output directory for final results
		"""

		start_time = time.time()

		self.benchmark_gt_dir = gt_dir
		self.seq_file =  "{}-{}.txt".format(benchmark_name, eval_mode)

		res_dir = res_dir
		self.benchmark_name = benchmark_name
		self.seqmaps_dir = seqmaps_dir

		self.mode = eval_mode

		self.datadir = os.path.join(gt_dir, self.mode)

		# getting names of sequences to evaluate
		error_traceback = ""
		assert self.mode in ["train", "test", "all"], "mode: %s not valid " %s

		print("Evaluating Benchmark: %s" % self.benchmark_name)

		# ======================================================
		# Handle evaluation
		# ======================================================



		# load list of all sequences
		sequences = np.genfromtxt(os.path.join(self.seqmaps_dir , self.seq_file), dtype='str', skip_header=True)


		self.gtfiles = []
		self.tsfiles = []
		self.sequences = []
		for seq in sequences:
			tsf = os.path.join( res_dir, "%s.txt" % seq)

			if path.exists(tsf) and os.stat(tsf).st_size:
				self.tsfiles.append(tsf)
				self.sequences.append(seq)

				gtf = os.path.join(self.benchmark_gt_dir, self.mode ,seq, 'gt/gt.txt')
				if path.exists(gtf):
					self.gtfiles.append(gtf)
				else:
					raise Exception("Ground Truth %s missing" % gtf)

			# else:
			# 	raise Exception("Result file %s missing" % tsf)


		print('Found {} ground truth files and {} test files.'.format(len(self.gtfiles), len(self.tsfiles)))
		print(self.tsfiles)
		if not self.tsfiles:
			return None, None

		self.MULTIPROCESSING = True
		MAX_NR_CORES = 10
		# set number of core for mutliprocessing
		if self.MULTIPROCESSING:
			self.NR_CORES = np.minimum( MAX_NR_CORES, len(self.tsfiles))
		try:

			""" run evaluation """
			results = self.eval()

			# calculate overall results
			results_attributes = self.Overall_Results.metrics.keys()

			for attr in results_attributes:
				""" accumulate evaluation values over all sequences """
				try:
					self.Overall_Results.__dict__[attr] = sum(obj.__dict__[attr] for obj in self.results)
				except:
					pass
			cache_attributes = self.Overall_Results.cache_dict.keys()
			for attr in cache_attributes:
				""" accumulate cache values over all sequences """
				try:
					self.Overall_Results.__dict__[attr] = self.Overall_Results.cache_dict[attr]['func']([obj.__dict__[attr] for obj in self.results])
				except:
					pass
			print("evaluation successful")

			# Compute clearmot metrics for overall and all sequences
			for res in self.results:
				res.compute_clearmot()
			self.Overall_Results.compute_clearmot()


			self.accumulate_df(type = "mail")
			self.failed = False
			error = None


		except Exception as e:
			print(str(traceback.format_exc()))
			print ("<br> Evaluation failed! <br>")

			error_traceback+= str(traceback.format_exc())
			self.failed = True
			self.summary = None

		end_time=time.time()

		self.duration = (end_time - start_time)/60.

		# ======================================================
		# Collect evaluation errors
		# ======================================================
		if self.failed:

			startExc = error_traceback.split("<exc>")
			error_traceback = [m.split("<!exc>")[0] for m in startExc[1:]]

			error = ""

			for err in error_traceback:
			    error+="Error: %s" % err


			print( "Error Message", error)
			self.error = error
			print("ERROR %s" % error)


		print ("Evaluation Finished")
		print("Your Results")
		print(self.render_summary())
		# save results if path set
		if save_pkl:


			self.Overall_Results.save_dict(os.path.join( save_pkl, "%s-%s-overall.pkl" % (self.benchmark_name, self.mode)))
			for res in self.results:
				res.save_dict(os.path.join( save_pkl, "%s-%s-%s.pkl" % (self.benchmark_name, self.mode, res.seqName)))
			print("Successfully save results")

		return self.Overall_Results, self.results
	def eval(self):
		raise NotImplementedError


	def accumulate_df(self, type = None):
		""" create accumulated dataframe with all sequences """
		for k, res in enumerate(self.results):
			res.to_dataframe(display_name = True, type = type )
			if k == 0: summary = res.df
			else: summary = summary.append(res.df)
		summary = summary.sort_index()


		self.Overall_Results.to_dataframe(display_name = True, type = type )

		self.summary = summary.append(self.Overall_Results.df)



	def render_summary( self, buf = None):
		"""Render metrics summary to console friendly tabular output.

		Params
		------
		summary : pd.DataFrame
		    Dataframe containing summaries in rows.

		Kwargs
		------
		buf : StringIO-like, optional
		    Buffer to write to
		formatters : dict, optional
		    Dicionary defining custom formatters for individual metrics.
		    I.e `{'mota': '{:.2%}'.format}`. You can get preset formatters
		    from MetricsHost.formatters
		namemap : dict, optional
		    Dictionary defining new metric names for display. I.e
		    `{'num_false_positives': 'FP'}`.

		Returns
		-------
		string
		    Formatted string
		"""

		if self.summary is None:
			return None

		output = self.summary.to_string(
			buf=buf,
			formatters=self.Overall_Results.formatters,
			justify = "left"
		)

		return output

def run_metrics( metricObject, args ):
	""" Runs metric for individual sequences
	Params:
	-----
	metricObject: metricObject that has computer_compute_metrics_per_sequence function
	args: dictionary with args for evaluation function
	"""
	metricObject.compute_metrics_per_sequence(**args)
	return metricObject


if __name__ == "__main__":
	Evaluator()