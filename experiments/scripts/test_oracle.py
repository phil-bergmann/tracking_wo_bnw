from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from sacred import Experiment
from model.config import cfg as frcnn_cfg
from model.config import cfg_from_list, cfg_from_file
import os
import os.path as osp
import yaml
import time

from tracker.rfrcnn import FRCNN as rFRCNN
from tracker.vfrcnn import FRCNN as vFRCNN
from tracker.config import get_output_dir
from tracker.utils import plot_sequence
from tracker.datasets.factory import Datasets
from tracker.oracle_tracker import Tracker
from tracker.utils import interpolate
from tracker.resnet import resnet50

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

ex = Experiment()

ex.add_config('experiments/cfgs/oracle_tracker.yaml')

# hacky workaround to load the corresponding configs and not having to hardcode paths here
ex.add_config(ex.configurations[0]._conf['oracle']['siamese_config'])
ex.add_config(ex.configurations[0]._conf['oracle']['frcnn_config'])

Tracker = ex.capture(Tracker, prefix='oracle.tracker')
    
@ex.automain
def my_main(oracle, siamese, frcnn, _config):
    # set all seeds
    torch.manual_seed(oracle['seed'])
    torch.cuda.manual_seed(oracle['seed'])
    np.random.seed(oracle['seed'])
    torch.backends.cudnn.deterministic = True

    if frcnn['cfg_file']:
        cfg_from_file(frcnn['cfg_file'])
    if frcnn['set_cfgs']:
        cfg_from_list(frcnn['set_cfgs'])

    print(_config)

    output_dir = osp.join(get_output_dir(oracle['module_name']), oracle['name'])
    
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')
    
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################
    
    print("[*] Building FRCNN")

    if oracle['network'] == 'vgg16':
        frcnn = vFRCNN()
    elif oracle['network'] == 'res101':
        frcnn = rFRCNN(num_layers=101)
    else:
        raise NotImplementedError("Network not understood: {}".format(oracle['network']))

    frcnn.create_architecture(2, tag='default',
        anchor_scales=frcnn_cfg.ANCHOR_SCALES,
        anchor_ratios=frcnn_cfg.ANCHOR_RATIOS)
    frcnn.eval()
    frcnn.cuda()
    frcnn.load_state_dict(torch.load(oracle['frcnn_weights']))
    
    cnn = resnet50(pretrained=False, **siamese['cnn'])
    cnn.load_state_dict(torch.load(oracle['siamese_weights']))
    cnn.eval()
    cnn.cuda()
    tr = Tracker(frcnn=frcnn, cnn=cnn)

    print("[*] Beginning evaluation...")

    time_ges = 0

    for db in Datasets(oracle['dataset']):
        tr.reset()

        now = time.time()

        print("[*] Evaluating: {}".format(db))

        dl = DataLoader(db, batch_size=1, shuffle=False)
        for sample in dl:
            tr.step(sample)
        results = tr.get_results()

        time_ges += time.time() - now

        print("Tracks found: {}".format(len(results)))
        print("[*] Time needed for {} evaluation: {:.3f} s".format(db, time.time() - now))

        if oracle['interpolate']:
            results = interpolate(results)

        db.write_results(results, osp.join(output_dir))
        
        if oracle['write_images']:
            plot_sequence(results, db, osp.join(output_dir, oracle['dataset'], str(db)))
    
    print("[*] Evaluation for all sets (without image generation): {:.3f} s".format(time_ges))
