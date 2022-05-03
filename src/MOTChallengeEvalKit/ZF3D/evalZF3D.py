import sys, os
sys.path.append(os.path.abspath(os.getcwd()))
import math
from collections import defaultdict
from Evaluator import Evaluator, run_metrics
import multiprocessing as mp
import pandas as pd


""" TODO import you metric object """
from ZF3D_metrics import Zef_3dMetrics


""" TODO define you evaluator """
class Zef_3d(Evaluator):
    def __init__(self):
        """ set type of your challenge """
        self.type = "ZF3D"

    def eval(self):
        """ check if files are complete and in correct format:
        e.g. Check for tracking if no duplicate ID/Frame is present in sequences
        """
        print("Check prediction files")
        error_message = ""
        for pred_file in self.tsfiles:
            df = pd.read_csv(pred_file, header = None)
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

        arguments = []
        # list of arguments for evaluation of each sequence
        for seq, res, gt in zip(self.sequences, self.tsfiles, self.gtfiles):
            """ write arguments for evaluation in list """
            det_df = pd.read_csv(res, sep=",", header=None, usecols=[0,1,2,3,4], names=['frame','id','3d_x','3d_y','3d_z'])
            gt_df = pd.read_csv(gt, sep=",", header=None, usecols=[0,1,2,3,4], names=['frame','id','3d_x','3d_y','3d_z'])
            arguments.append({"metricObject": Zef_3dMetrics(seq),
                              "args" : {
                              "sequence":seq,
                              "det_df": det_df,
                              "gt_df": gt_df,
                              "gtDataDir":  os.path.join( self.datadir,seq),
                              "benchmark_name": self.benchmark_name,
                             }
                             })
        try:
            if self.MULTIPROCESSING:
                """ start multiprocessing evaluation """
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

            raise Exception("<exc> Evaluation failed <!exc>")

        self.Overall_Results = Zef_3dMetrics("OVERALL")

if __name__ == "__main__":

    eval = Zef_3d()

    benchmark_name = "3D-ZeF20"
    gt_dir = "data/3DZeF20"
    res_dir = "res/ZeFres"
    eval_mode = "train"

    eval.run(
		benchmark_name = benchmark_name,
		gt_dir = gt_dir,
		res_dir = res_dir,
		eval_mode = eval_mode,
		save_pkl = "eval_results")
