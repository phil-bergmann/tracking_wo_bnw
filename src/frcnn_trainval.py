# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import pprint
import sys

import numpy as np
import torch

from frcnn.datasets import imdb
from frcnn.datasets.factory import get_imdb
from frcnn.model.config import (cfg, cfg_from_file, cfg_from_list,
                                get_output_dir, get_output_tb_dir)
from frcnn.model.train_val import get_training_roidb, train_net
from frcnn.nets.resnet_v1 import resnetv1
from frcnn.nets.vgg16 import vgg16


def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb


def frcnn_trainval(imdb_name, imdbval_name, max_iters, pretrained_model, pretrained_full_model, cfg_file, set_cfgs, network, tag):
  """
  args = {'imdb_name':imdb_name,
      'imdbval_name':imdbval_name,
      'max_iters':max_iters,
      'net':network,
      'cfg_file':cfg_file,
      'set_cfgs':set_cfgs,
      'weights':weights,
      'tag':tag}
  """

  if cfg_file:
    cfg_from_file(cfg_file)
  if set_cfgs:
    cfg_from_list(set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  np.random.seed(cfg.RNG_SEED)

  # train set
  imdb, roidb = combined_roidb(imdb_name)
  print('{:d} roidb entries'.format(len(roidb)))

  # output directory where the models are saved
  output_dir = get_output_dir(imdb, tag)
  print('Output will be saved to `{:s}`'.format(output_dir))

  # tensorboard directory where the summaries are saved during training
  tb_dir = get_output_tb_dir(imdb, tag)
  print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  #-, valroidb = combined_roidb(args['imdbval_name'])
  valimdb, valroidb = combined_roidb(imdbval_name)
  print('{:d} validation roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip

  # load network
  if network == 'vgg16':
    net = vgg16()
  elif network == 'res50':
    net = resnetv1(num_layers=50)
  elif network == 'res101':
    net = resnetv1(num_layers=101)
  elif network == 'res152':
    net = resnetv1(num_layers=152)
  elif network == 'mobile':
    net = mobilenetv1()
  else:
    raise NotImplementedError


  train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
            pretrained_model=pretrained_model,
            pretrained_full_model=pretrained_full_model,
            max_iters=max_iters)
