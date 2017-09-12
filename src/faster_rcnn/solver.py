import os
from random import shuffle
import numpy as np
from datetime import datetime

import torch
from torch.autograd import Variable

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

try:
    from termcolor import cprint
except ImportError:
    cprint = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

class Solver(object):

    def __init__(self):

        self.pretrained_model = 'models/VGG_imagenet.npy'
        self.cfg_file = 'cfg/faster_rcnn_end2end.yml'
        self.output_dir = 'models/saved_model3'

    def train(self, imdb_name = 'voc_2007_trainval'):
        """
        Train a given model with the provided data.

        """

        # hyper-parameters
        # ------------
        start_step = 0
        end_step = 100000
        lr_decay_steps = {60000, 80000}
        lr_decay = 1./10
        # ------------

        np.random.seed(1024)

        # load config
        cfg_from_file(self.cfg_file)
        lr = cfg.TRAIN.LEARNING_RATE
        momentum = cfg.TRAIN.MOMENTUM
        weight_decay = cfg.TRAIN.WEIGHT_DECAY
        disp_interval = cfg.TRAIN.DISPLAY
        log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

        # load data
        imdb = get_imdb(imdb_name)
        rdl_roidb.prepare_roidb(imdb)
        roidb = imdb.roidb
        data_layer = RoIDataLayer(roidb, imdb.num_classes)

        # load net
        net = FasterRCNN(classes=imdb.classes)
        network.weights_normal_init(net, dev=0.01)
        network.load_pretrained_npy(net, self.pretrained_model, encoding='latin1')

        net.cuda()
        net.train()

        params = list(net.parameters())
        optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        print('START TRAIN.')

        # training
        train_loss = 0
        tp, tf, fg, bg = 0., 0., 0, 0
        step_cnt = 0
        re_cnt = False
        t = Timer()
        t.tic()

        for step in range(end_step):
            # get one batch
            blobs = data_layer.forward()
            im_data = blobs['data']
            im_info = blobs['im_info']
            gt_boxes = blobs['gt_boxes']
            gt_ishard = blobs['gt_ishard']
            dontcare_areas = blobs['dontcare_areas']

            # forward
            net(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
            loss = net.loss + net.rpn.loss

            train_loss += loss.data[0]
            step_cnt += 1

            # backward
            optimizer.zero_grad()
            loss.backward()
            network.clip_gradient(net, 10.)
            optimizer.step()

            if step % disp_interval == 0:
                duration = t.toc(average=False)
                fps = step_cnt / duration

                log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
                    step, blobs['im_name'], train_loss / step_cnt, fps, 1./fps)
                log_print(log_text, color='green', attrs=['bold'])

                re_cnt = True

            if (step % 10000 == 0) and step > 0:
                save_name = os.path.join(self.output_dir, 'faster_rcnn_{}.h5'.format(step))
                network.save_net(save_name, net)
                print(('save model: {}'.format(save_name)))

            if step in lr_decay_steps:
                lr *= lr_decay
                optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

            if re_cnt:
                tp, tf, fg, bg = 0., 0., 0, 0
                train_loss = 0
                step_cnt = 0
                t.tic()
                re_cnt = False
            
        print('FINISH.')
