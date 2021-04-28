import copy
import os
import time
from os import path as osp

import motmetrics as mm
import numpy as np
import sacred
import torch
import yaml
from sacred import Experiment
from torch.utils.data import DataLoader
from tqdm import tqdm
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.oracle_tracker import OracleTracker
from tracktor.reid.resnet import ReIDNetwork_resnet50
from tracktor.tracker import Tracker
from tracktor.utils import (evaluate_mot_accums, get_mot_accum,
                            interpolate_tracks, plot_sequence)

mm.lap.default_solver = 'lap'


ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')
ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')


@ex.config
def add_reid_config(reid_models, obj_detect_models, dataset):
    if isinstance(reid_models, str):
        reid_models = [reid_models, ] * len(dataset)

    # if multiple reid models are provided each is applied
    # to a different dataset
    if len(reid_models) > 1:
        assert len(dataset) == len(reid_models)

    # reid_cfgs = []
    # for reid_model in reid_models:
    #     reid_config = os.path.join(
    #         os.path.dirname(reid_model),
    #         'sacred_config.yaml')
    #     reid_cfgs.append(yaml.safe_load(open(reid_config)))

    if isinstance(obj_detect_models, str):
        obj_detect_models = [obj_detect_models, ] * len(dataset)
    if len(obj_detect_models) > 1:
        assert len(dataset) == len(obj_detect_models)


@ex.automain
def main(module_name, name, seed, obj_detect_models, reid_models,
         tracker, oracle, dataset, load_results, frame_range, interpolate,
         write_images, _config, _log, _run):
    sacred.commands.print_config(_run)

    # set all seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    output_dir = osp.join(get_output_dir(module_name), name)
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(copy.deepcopy(_config), outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    _log.info("Initializing object detector(s).")

    obj_detects = []
    for obj_detect_model in obj_detect_models:
        obj_detect = FRCNN_FPN(num_classes=2)
        obj_detect.load_state_dict(torch.load(obj_detect_model,
                                map_location=lambda storage, loc: storage))
        obj_detects.append(obj_detect)

        obj_detect.eval()
        if torch.cuda.is_available():
            obj_detect.cuda()

    # reid
    _log.info("Initializing reID network(s).")

    reid_networks = []
    for reid_model in reid_models:
        reid_cfg = os.path.join(os.path.dirname(reid_model), 'sacred_config.yaml')
        reid_cfg = yaml.safe_load(open(reid_cfg))

        reid_network = ReIDNetwork_resnet50(pretrained=False, **reid_cfg['model_args'])
        reid_network.load_state_dict(torch.load(reid_model,
                                    map_location=lambda storage, loc: storage))
        reid_network.eval()
        if torch.cuda.is_available():
            reid_network.cuda()

        reid_networks.append(reid_network)

    # tracktor
    if oracle is not None:
        tracker = OracleTracker(
            obj_detect, reid_network, tracker, oracle)
    else:
        tracker = Tracker(obj_detect, reid_network, tracker)

    time_total = 0
    num_frames = 0
    mot_accums = []
    dataset = Datasets(dataset)

    for seq, obj_detect, reid_network in zip(dataset, obj_detects, reid_networks):
        tracker.obj_detect = obj_detect
        tracker.reid_network = reid_network
        tracker.reset()

        _log.info(f"Tracking: {seq}")

        start_frame = int(frame_range['start'] * len(seq))
        end_frame = int(frame_range['end'] * len(seq))

        seq_loader = DataLoader(torch.utils.data.Subset(seq, range(start_frame, end_frame)))
        num_frames += len(seq_loader)

        results = {}
        if load_results:
            results = seq.load_results(output_dir)
        if not results:
            start = time.time()

            for frame_data in tqdm(seq_loader):
                with torch.no_grad():
                    tracker.step(frame_data)

            results = tracker.get_results()

            time_total += time.time() - start

            _log.info(f"Tracks found: {len(results)}")
            _log.info(f"Runtime for {seq}: {time.time() - start :.2f} s.")

            if interpolate:
                results = interpolate_tracks(results)

            _log.info(f"Writing predictions to: {output_dir}")
            seq.write_results(results, output_dir)

        if seq.no_gt:
            _log.info("No GT data for evaluation available.")
        else:
            mot_accums.append(get_mot_accum(results, seq_loader))

        if write_images:
            plot_sequence(
                results,
                seq,
                osp.join(output_dir, str(dataset), str(seq)),
                write_images)

    if time_total:
        _log.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
                f"{time_total:.2f} s for {num_frames} frames ({num_frames / time_total:.2f} Hz)")
    if mot_accums:
        _log.info("Evaluation:")
        evaluate_mot_accums(mot_accums,
                            [str(s) for s in dataset if not s.no_gt],
                            generate_overall=True)
