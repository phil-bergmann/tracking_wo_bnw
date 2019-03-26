from __future__ import absolute_import, division, print_function

import os
import os.path as osp

# Change paths in config file
from frcnn.model import config
from frcnn.model.config import cfg, cfg_from_file, cfg_from_list
from frcnn_test import frcnn_test
from sacred import Experiment

this_dir = osp.dirname(__file__)
root_path = osp.join(this_dir, '..', '..')
data_path = osp.join(root_path, 'data')
config.cfg_from_list(['ROOT_DIR', root_path, 'DATA_DIR', data_path])


ex = Experiment()

frcnn_test = ex.capture(frcnn_test)


@ex.config
def default():
    score_thresh = 0.05
    nms_thresh = 0.3
    clip_bbox = False
    max_per_image = 100
    output_name = None
    write_images = False

    # Added so that sacred doesn't throw a key error
    #description = ""
    #timestamp = ""
    #imdbval_name = ""
    #weights = ""
    #network = ""

# Dataset configs
@ex.named_config
def small_mot():
    imdbtest_name = "mot_2017_small_val"

@ex.named_config
def mot():
    imdbtest_name = "mot_2017_all"

@ex.named_config
def mot_test():
    imdbtest_name = "mot_2017_test"

@ex.named_config
def mot_train():
    imdbtest_name = "mot_2017_train"

@ex.named_config
def kitti_car_train():
    imdbtest_name = "kitti_detection_Car_train"

@ex.named_config
def kitti_car_test():
    imdbtest_name = "kitti_detection_Car_test"

@ex.named_config
def kitti_car_small_val():
    imdbtest_name = "kitti_detection_Car_small_val"

@ex.named_config
def kitti_pedestrian_train():
    imdbtest_name = "kitti_detection_Pedestrian_train"

@ex.named_config
def kitti_pedestrian_test():
    imdbtest_name = "kitti_detection_Pedestrian_test"

@ex.named_config
def kitti_pedestrian_small_val():
    imdbtest_name = "kitti_detection_Pedestrian_small_val"


@ex.automain
def my_main(imdbtest_name, clip_bbox, output_name, nms_thresh, frcnn, write_images, _config):

    # Clip bboxes after bbox reg to image boundary
    cfg_from_list(['TEST.BBOX_CLIP', str(clip_bbox), 'TEST.NMS', str(nms_thresh)])
    #cfg_from_list(["APPLY_CLAHE", "True"])

    # Already set everything here, so the path can be determined correctly
    if frcnn['cfg_file']:
        cfg_from_file(frcnn['cfg_file'])
    if frcnn['set_cfgs']:
        cfg_from_list(frcnn['set_cfgs'])

    model_dir = osp.abspath(osp.join(cfg.ROOT_DIR, 'output', 'frcnn', cfg.EXP_DIR,
        frcnn['imdb_name'], frcnn['tag']))
    model = osp.join(model_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(frcnn['max_iters']) + '.pth')
    # model = osp.join(model_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_35000' + '.pth')
    if output_name:
        output_dir = osp.join(model_dir, output_name)
    else:
        output_dir = osp.join(model_dir, imdbtest_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Called with args:')
    print(_config)

    frcnn_test(model=model, output_dir=output_dir, network=frcnn['network'],
               write_images=write_images)
