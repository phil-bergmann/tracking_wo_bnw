from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
__C.TRAIN = edict()

# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0005

# Number of iterations that are used to average the error
__C.TRAIN.ERROR_AVERAGE_ITERATIONS = 100

# Samples to use per minibatch
__C.TRAIN.SMP_PER_BATCH = 1

# Snapshot prefix
__C.TRAIN.SNAPSHOT_PREFIX = "LSTM_v0.1"

# Make Snapshot every Iters
__C.TRAIN.SNAPSHOT_ITERS = 1000


#
# Testing options
#
__C.TEST = edict()


#
# LSTM settings
#
__C.LSTM = edict()

# number of hidden neurons
__C.LSTM.HIDDEN_NUM = 500

# number of layers
__C.LSTM.LAYERS = 1




# For reproducibility
__C.RNG_SEED = 3

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# 



def get_output_dir(module):
  """Return the directory where experimental artifacts are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', 'tracker', module))
  #if weights_filename is None:
  #  weights_filename = 'default'
  #outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir

def get_tb_dir(module):
  """Return the directory where experimental artifacts are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'tensorboard', 'tracker', module))
  #if weights_filename is None:
  #  weights_filename = 'default'
  #outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


