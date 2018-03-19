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


from tracker.sfrcnn import FRCNN
from tracker.config import cfg, get_output_dir
from tracker.utils import plot_sequence
from tracker.mot_sequence import MOT_Sequence
from tracker.frcnn_tracker import FRCNN_Tracker

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

ex = Experiment()

ex.add_config('experiments/cfgs/simple_tracker.yaml')
FRCNN_Tracker = ex.capture(FRCNN_Tracker, prefix='simple_tracker.tracker')

test = ["MOT17-01", "MOT17-03", "MOT17-06", "MOT17-07", "MOT17-08", "MOT17-12", "MOT17-14"]
train = ["MOT17-13", "MOT17-11", "MOT17-10", "MOT17-09", "MOT17-05", "MOT17-04", "MOT17-02", ]
sequences = ["MOT17-09"]
sequences = train
    
@ex.automain
def my_main(simple_tracker, _config):
    # set all seeds
    torch.manual_seed(simple_tracker['seed'])
    torch.cuda.manual_seed(simple_tracker['seed'])
    np.random.seed(simple_tracker['seed'])
    torch.backends.cudnn.deterministic = True

    print(_config)

    output_dir = osp.join(get_output_dir(simple_tracker['module_name']), simple_tracker['name'])
    
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')
    
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################
    print("[*] Building FRCNN")

    frcnn = FRCNN()
    frcnn.create_architecture(2, tag='default',
        anchor_scales=frcnn_cfg.ANCHOR_SCALES,
        anchor_ratios=frcnn_cfg.ANCHOR_RATIOS)
    frcnn.eval()
    frcnn.cuda()
    frcnn.load_state_dict(torch.load(simple_tracker['frcnn_weights']))
    
    tracker = FRCNN_Tracker(frcnn)

    print("[*] Beginning evaluation...")

    time_ges = 0

    for s in sequences:
        tracker.reset()

        now = time.time()

        print("[*] Evaluating: {}".format(s))
        db = MOT_Sequence(s)
        dl = DataLoader(db, batch_size=1, shuffle=False)
        for sample in dl:
            tracker.step(sample)
        results = tracker.get_results()

        time_ges += time.time() - now

        print("Tracks found: {}".format(len(results)))
        print("[*] Time needed for {} evaluation: {:.3f} s".format(s, time.time() - now))

        db.write_results(results, osp.join(output_dir))
        
        plot_sequence(results, db, osp.join(output_dir, s))
    
    print("[*] Evaluation for all sets (without image generation): {:.3f} s".format(time_ges))