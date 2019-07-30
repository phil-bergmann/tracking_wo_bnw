# --------------------------------------------------------
# Pytorch FPN implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang, based on code from faster R-CNN
# --------------------------------------------------------

from __future__ import absolute_import, division, print_function

import argparse
import os
import pdb
import pickle
import pprint
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import cv2
from fpn.model.fpn.resnet import FPNResNet
from fpn.model.nms.nms_wrapper import nms, soft_nms
from fpn.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from fpn.model.utils.config import (cfg, cfg_from_file, cfg_from_list,
                                    get_output_dir)
from fpn.model.utils.net_utils import vis_detections
from fpn.roi_data_layer.roibatchLoader import roibatchLoader
from fpn.roi_data_layer.roidb import combined_roidb
from fpn.test import validate


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('exp_name', type=str, default=None, help='experiment name')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--imdbval_name', dest='imdbval_name',
                        help='val imdb',
                        default='', type=str)
    # parser.add_argument('--cfg', dest='cfg_file',
    #                     help='optional config file',
    #                     default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="output/fpn",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--score_thresh', dest='score_thresh',
                        help='treshhold for classification score',
                        default=0.05, type=float)

    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--soft_nms', help='whether use soft_nms', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.dataset == "pascal_voc":
        args.imdbval_name = "voc_2007_test"
        set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdbval_name = "voc_0712_test"
        set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdbval_name = "coco_2014_minival"
        set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdbval_name = "imagenet_val"
        set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdbval_name = "vg_150-50-50_minival"
        set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif 'mot_2017' in args.dataset or 'mot19_cvpr' in args.dataset:
        set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    # elif args.dataset == "mot_2017_train":
    #     # args.imdb_name = "mot_2017_train"
    #     args.imdbval_name = "mot_2017_train"
    #     set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    # elif args.dataset == "mot_2017_small_train":
    #     # args.imdb_name = "mot_2017_small_train"
    #     args.imdbval_name = "mot_2017_small_train"
    #     set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    # elif args.dataset == "mot_2017_small_val":
    #     # args.imdb_name = "mot_2017_small_train"
    #     args.imdbval_name = "mot_2017_small_val"
    #     set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    # elif args.dataset == "mot_2017_all":
    #     # args.imdb_name = "mot_2017_small_train"
    #     args.imdbval_name = "mot_2017_all"
    #     set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    else:
        raise NotImplementedError

    input_dir = os.path.join(args.load_dir, args.net,
                             args.dataset, args.exp_name)
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(
        input_dir, f"fpn_{args.checksession}_{args.checkepoch}.pth")

    cfg_file = os.path.join(input_dir, 'config.yaml')
    cfg_from_file(cfg_file)
    cfg_from_list(set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # data
    # cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)
    print('{:d} roidb entries'.format(len(roidb)))
    output_dir = get_output_dir(imdb, args.exp_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1,
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=4,
                                             pin_memory=True)

    # network
    print("load checkpoint %s" % (load_name))

    if args.net == 'res101':
        FPN = FPNResNet(imdb.classes, 101, pretrained=False)
    elif args.net == 'res50':
        FPN = FPNResNet(imdb.classes, 50, pretrained=False)
    elif args.net == 'res152':
        FPN = FPNResNet(imdb.classes, 152, pretrained=False)
    else:
        print("Network is not defined.")
        pdb.set_trace()

    FPN.create_architecture()
    FPN.load_state_dict(torch.load(load_name)['model'])
    print('load model successfully!')

    if args.cuda:
        cfg.CUDA = True
        FPN.cuda()
    elif torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    start = time.time()

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    all_boxes = validate(FPN, dataloader, imdb, vis=args.vis,
                         cuda=args.cuda, soft_nms=args.soft_nms, score_thresh=args.score_thresh)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    end = time.time()
    print("test time: %0.4fs" % (end - start))


if __name__ == '__main__':
    main()
