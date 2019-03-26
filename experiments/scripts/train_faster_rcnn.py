from __future__ import absolute_import, division, print_function

import os
import os.path as osp
from datetime import datetime

import yaml

# Change paths in config file
from frcnn.model import config
from frcnn.model.config import cfg, cfg_from_file, cfg_from_list
from frcnn_trainval import frcnn_trainval
from sacred import Experiment

this_dir = osp.dirname(__file__)
root_path = osp.join(this_dir, '..', '..')
data_path = osp.join(root_path, 'data')
config.cfg_from_list(['ROOT_DIR', root_path, 'DATA_DIR', data_path])


ex = Experiment()

frcnn_trainval = ex.capture(frcnn_trainval)


@ex.config
def default():
    set_cfgs = None
    cfg_file = None
    tag =  datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    description = ""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    pretrained_model = None
    pretrained_full_model = None

# Dataset configs
@ex.named_config
def small_mot():
    imdb_name = "mot_2017_small_train"
    imdbval_name = "mot_2017_small_val"
    max_iters = 180000
    set_cfgs = ["TRAIN.STEPSIZE", "[125000]"]

@ex.named_config
def mot():
    imdb_name = "mot_2017_train"
    imdbval_name = "mot_2017_small_val"
    max_iters = 180000
    set_cfgs = ["TRAIN.STEPSIZE", "[125000]"]

@ex.named_config
def small_kitti_car():
    imdb_name = "kitti_detection_Car_small_train"
    imdbval_name = "kitti_detection_Car_small_val"
    max_iters = 180000
    set_cfgs = ["TRAIN.STEPSIZE", "[125000]"]

@ex.named_config
def kitti_car():
    imdb_name = "kitti_detection_Car_train"
    imdbval_name = "kitti_detection_Car_small_val"
    max_iters = 180000
    set_cfgs = ["TRAIN.STEPSIZE", "[125000]"]

@ex.named_config
def small_kitti_pedestrian():
    imdb_name = "kitti_detection_Pedestrian_small_train"
    imdbval_name = "kitti_detection_Pedestrian_small_val"
    #max_iters = 180000
    #set_cfgs = ["TRAIN.STEPSIZE", "[125000]"]
    max_iters = 100000
    set_cfgs = ["TRAIN.STEPSIZE", "[35000]"]
    #set_cfgs += ["TRAIN.LEARNING_RATE", "0.0001", "TRAIN.WEIGHT_DECAY", "0.00001"]
    # make shorter side similar to MOT size
    set_cfgs += ["TRAIN.MAX_SIZE", "1450", "TEST.MAX_SIZE", "1450"]
    #set_cfgs += ["TRAIN.SNAPSHOT_KEPT", "20"]
    # contrast equalization
    #set_cfgs += ["APPLY_CLAHE", "True"]

@ex.named_config
def kitti_pedestrian():
    imdb_name = "kitti_detection_Pedestrian_train"
    imdbval_name = "kitti_detection_Pedestrian_small_val"
    max_iters = 100000
    set_cfgs = ["TRAIN.STEPSIZE", "[35000]"]
    # make shorter side similar to MOT size
    set_cfgs += ["TRAIN.MAX_SIZE", "1450", "TEST.MAX_SIZE", "1450"]
    #set_cfgs += ["TRAIN.SNAPSHOT_KEPT", "20"]

@ex.named_config
def mot_pretrained():
    pretrained_full_model = "output/frcnn/res101/mot_2017_train/STOP180k_NEW/res101_faster_rcnn_iter_180000.pth"

@ex.named_config
def res50():
    network = "res50"
    pretrained_model = "data/imagenet_weights/{}.pth".format(network)
    cfg_file = "experiments/cfgs/{}.yml".format(network)

@ex.named_config
def res101():
    network = "res101"
    pretrained_model = "data/imagenet_weights/{}.pth".format(network)
    cfg_file = "experiments/cfgs/{}.yml".format(network)

@ex.named_config
def vgg16():
    network = "vgg16"
    pretrained_model = "data/imagenet_weights/{}.pth".format(network)
    cfg_file = "experiments/cfgs/{}.yml".format(network)


@ex.automain
def my_main(tag, cfg_file, set_cfgs, imdb_name, _config):

    # Already set everything here, so the path can be determined correctly
    if cfg_file:
        cfg_from_file(cfg_file)
    if set_cfgs:
        cfg_from_list(set_cfgs)

    print('Called with args:')
    print(_config)

    # if not already present save the configuration into a file in the output folder
    outdir = osp.abspath(osp.join(cfg.ROOT_DIR, 'output', 'frcnn', cfg.EXP_DIR, imdb_name, tag))
    sacred_config = osp.join(outdir, 'sacred_config.yaml')
    if not osp.isfile(sacred_config):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with open(sacred_config, 'w') as outfile:
            yaml.dump({'frcnn':_config}, outfile, default_flow_style=False)

    frcnn_trainval()
