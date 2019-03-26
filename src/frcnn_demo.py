#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import, division, print_function

import os
import os.path as osp
import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import cv2
from frcnn.datasets.factory import get_imdb
from frcnn.model.config import cfg
from frcnn.model.nms_wrapper import nms
from frcnn.model.test import im_detect
from frcnn.nets.resnet_v1 import resnetv1
from frcnn.nets.vgg16 import vgg16
from frcnn.utils.timer import Timer

matplotlib.use('Agg')

#CLASSES = ('__background__',
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')

def vis_detections(im, class_name, dets, im_output, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.2f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(im_output)

def demo(net, image_name, imdb, output_dir, thresh):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    im_output = os.path.join(output_dir, image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    for cls_ind, cls in enumerate(imdb.classes[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(torch.from_numpy(dets), cfg.TEST.NMS)
        dets = dets[keep.numpy(), :]
        vis_detections(im, cls, dets, im_output, thresh=thresh)

def frcnn_demo(args):

    if args['cfg_file']:
        cfg_from_file(args['cfg_file'])
    if args['set_cfgs']:
        cfg_from_list(args['set_cfgs'])

    #cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    print('Using config:')
    pprint.pprint(cfg)

    imdb = get_imdb(args['imdb_name'])

    # load network
    if args['net'] == 'vgg16':
        net = vgg16()
    elif args['net'] == 'res50':
        net = resnetv1(num_layers=50)
    elif args['net'] == 'res101':
        net = resnetv1(num_layers=101)
    elif args['net'] == 'res152':
        net = resnetv1(num_layers=152)
    else:
        raise NotImplementedError

    # load model
    net.create_architecture(imdb.num_classes, tag='default',
        anchor_scales=cfg.ANCHOR_SCALES,
        anchor_ratios=cfg.ANCHOR_RATIOS)

    net.load_state_dict(torch.load(args['model']))

    net.eval()
    net.cuda()

    print('Loaded network {:s}'.format(args['model']))

    for im_name in args['im_names']:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(net, im_name, imdb, args['output_dir'], thresh=args['score_thresh'])

    #plt.savefig(osp.join(args['output_dir'], 'test.png'))
