# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import argparse
import os
import pprint
import sys
import time
from os import path as osp

import matplotlib.pyplot as plt
import torch

import cv2
from frcnn.datasets.factory import get_imdb
from frcnn.model.config import cfg, cfg_from_file, cfg_from_list
from frcnn.model.test import test_net
from frcnn.nets.resnet_v1 import resnetv1
from frcnn.nets.vgg16 import vgg16


def frcnn_test(imdbtest_name, network, model, output_dir, score_thresh,
               max_per_image, write_images=False):
    """
    args = {#'imdb_name':imdb_name,
            'imdbtest_name':imdbtest_name,
            'net':network,
            'cfg_file':None,
            'set_cfgs':None,
            'tag':tag,
            'comp_mode':comp,
            'max_per_image':max_per_image,
            'output_dir':output_dir,
            'model':model}
    """

    #if cfg_file:
    #    cfg_from_file(cfg_file)
    #if set_cfgs:
    #    cfg_from_list(set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    #filename = os.path.splitext(os.path.basename(args['model']))[0]

    imdb_test = get_imdb(imdbtest_name)
    #imdb_test.competition_mode(args['comp_mode'])

    # load network
    if network == 'vgg16':
        net = vgg16()
    elif network == 'res50':
        net = resnetv1(num_layers=50)
    elif network == 'res101':
        net = resnetv1(num_layers=101)
    elif network == 'res152':
        net = resnetv1(num_layers=152)
    else:
        raise NotImplementedError

    # load model
    net.create_architecture(imdb_test.num_classes, tag='default',
        anchor_scales=cfg.ANCHOR_SCALES,
        anchor_ratios=cfg.ANCHOR_RATIOS)

    net.eval()

    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(count_parameters(net))
    # exit()
    net.cuda()

    print(('Loading model check point from {:s}').format(model))
    net.load_state_dict(torch.load(model))
    print('Loaded.')

    all_boxes = test_net(net, imdb_test, output_dir, max_per_image=max_per_image, thresh=score_thresh)

    if write_images:
        num_images = len(imdb_test.image_index)

        for i in range(num_images):
            im_path = imdb_test.image_path_at(i)
            im = cv2.imread(im_path)
            im = im[:, :, (2, 1, 0)]

            fig, ax = plt.subplots(1,1)
            ax.imshow(im, aspect='equal')

            for t_i in all_boxes[1][i]:
                ax.add_patch(
                plt.Rectangle((t_i[0], t_i[1]),
                          t_i[2] - t_i[0],
                          t_i[3] - t_i[1], fill=False,
                          linewidth=1.0)
                )

            im_name = osp.basename(im_path)
            im_output = osp.join(output_dir, im_name)
            plt.axis('off')
            plt.tight_layout()
            plt.draw()
            plt.savefig(im_output)
            plt.close()
