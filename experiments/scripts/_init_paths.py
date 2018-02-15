from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# setup matlpoltib to use without display
import matplotlib
matplotlib.use('Agg')

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

root_path = osp.join(this_dir, '..', '..')

# Add src to PYTHONPATH
src_path = osp.join(root_path, 'src')
add_path(src_path)

# Change paths in config file
import frcnn
from model.config import cfg_from_list
data_path = osp.join(root_path, 'data')
cfg_from_list(['ROOT_DIR', root_path,'DATA_DIR', data_path])