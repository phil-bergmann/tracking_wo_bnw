import copy
import os
import os.path as osp
import random

import numpy as np
import sacred
import torch
import yaml
from torch.utils.data import DataLoader
from tracktor.config import get_output_dir, get_tb_dir
from tracktor.datasets.factory import Datasets
from tracktor.reid.resnet import ReIDNetwork_resnet50
from tracktor.reid.solver import Solver

ex = sacred.Experiment()
ex.add_config('experiments/cfgs/reid.yaml')


@ex.config
def add_dataset_to_model(model_args, dataset_kwargs):
    model_args.update({
        'crop_H': dataset_kwargs['crop_H'],
        'crop_W': dataset_kwargs['crop_W'],
        'normalize_mean': dataset_kwargs['normalize_mean'],
        'normalize_std': dataset_kwargs['normalize_std']})


@ex.automain
def main(seed, module_name, name, db_train, db_val, solver_cfg,
         model_args, dataset_kwargs, _run, _config, _log):
    # set all seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    sacred.commands.print_config(_run)

    output_dir = osp.join(get_output_dir(module_name), name)
    tb_dir = osp.join(get_tb_dir(module_name), name)

    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(copy.deepcopy(_config), outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################
    _log.info("[*] Building CNN")
    model = ReIDNetwork_resnet50(
        pretrained=True, **model_args)
    model.train()
    model.cuda()

    #########################
    # Initialize dataloader #
    #########################
    _log.info("[*] Initializing Datasets")

    _log.info("[*] Train:")
    dataset_kwargs = copy.deepcopy(dataset_kwargs)
    dataset_kwargs['logger'] = _log.info
    dataset_kwargs['mot_dir'] = db_train['mot_dir']
    dataset_kwargs['transform'] = db_train['transform']
    dataset_kwargs['random_triplets'] = db_train['random_triplets']

    db_train = Datasets(db_train['split'], dataset_kwargs)
    db_train = DataLoader(db_train, batch_size=1, shuffle=True)

    if db_val is not None:
        _log.info("[*] Val:")

        dataset_kwargs['mot_dir'] = db_val['mot_dir']
        dataset_kwargs['transform'] = db_val['transform']
        dataset_kwargs['random_triplets'] = db_val['random_triplets']
        db_val = Datasets(db_val['split'], dataset_kwargs)
        db_val = DataLoader(db_val, batch_size=1, shuffle=False)

    ##################
    # Begin training #
    ##################
    _log.info("[*] Solving ...")

    # build scheduling like in "In Defense of the Triplet Loss
    # for Person Re-Identification" from Hermans et al.
    def lr_scheduler(epoch):
        if epoch < 1 / 2 * solver_cfg['num_epochs']:
            return 1
        return 0.001 ** (2 * epoch / solver_cfg['num_epochs'] - 1)
        # return 0.1 ** (epoch // 30)
        # return 0.9 ** epoch

    solver = Solver(
        output_dir, tb_dir,
        lr_scheduler_lambda=lr_scheduler,
        logger=_log.info,
        optim=solver_cfg['optim'],
        optim_args=solver_cfg['optim_args'])
    solver.train(
        model, db_train, db_val, solver_cfg['num_epochs'], solver_cfg['log_nth'])
