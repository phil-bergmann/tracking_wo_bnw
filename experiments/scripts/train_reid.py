import argparse
import os
import os.path as osp
import sys
import time

import torch
import torch.nn as nn
import torchreid
from torchreid.data.datasets import __image_datasets
from torchreid.utils import (Logger, check_isfile, collect_env_info,
                             compute_model_complexity, load_pretrained_weights,
                             resume_from_checkpoint, set_random_seed)
from tracktor.reid.config import (check_cfg, engine_run_kwargs,
                                  get_default_config, lr_scheduler_kwargs,
                                  optimizer_kwargs, reset_config)
from tracktor.reid.datamanager import build_datamanager
from tracktor.reid.engine import build_engine
from tracktor.reid.mot_seq_dataset import get_sequence_class


def update_datasplits(cfg):
    assert isinstance(cfg.data.sources, (tuple, list))
    assert isinstance(cfg.data.sources, (tuple, list))

    if isinstance(cfg.data.sources[0], (tuple, list)):
        assert len(cfg.data.sources) == 1
        cfg.data.sources = cfg.data.sources[0]

    if isinstance(cfg.data.targets[0], (tuple, list)):
        assert len(cfg.data.targets) == 1
        cfg.data.targets = cfg.data.targets[0]


def register_datasets(datasets, cfg):
    if not isinstance(datasets, (tuple, list)):
        datasets = [datasets]

    for seq_name in datasets:
        print("Registering dataset ", seq_name)
        if seq_name not in __image_datasets:
            seq_class = get_sequence_class(seq_name, cfg)
            torchreid.data.register_image_dataset(seq_name, seq_class)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        nargs='+',
        help='source datasets (delimited by space)'
    )
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        nargs='+',
        help='target datasets (delimited by space)'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation'
    )
    parser.add_argument(
        '--root', type=str, default='', help='path to data root'
    )
    parser.add_argument(
        '--root-targets', type=str, default='', help='path to data root targets'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)
    check_cfg(cfg)

    if cfg.test.deid:
        assert cfg.test.evaluate, 'De-identifaction must be run with cfg.test.evaluate=True.'

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    update_datasplits(cfg)
    register_datasets(cfg.data.sources, cfg)
    register_datasets(cfg.data.targets, cfg)

    datamanager = build_datamanager(cfg)

    print(f'Building model: {cfg.model.name}')
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    num_params, flops = compute_model_complexity(
        model, (1, 3, cfg.data.height, cfg.data.width)
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    print(
        'Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type)
    )
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
    engine.run(**engine_run_kwargs(cfg))


if __name__ == '__main__':
    main()
