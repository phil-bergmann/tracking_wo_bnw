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
        pass

    def train(self, imdb_name='voc_2007_trainval', end_step=100000, lr_decay_steps={60000, 80000},
        output_dir='models/faster_rcnn', pretrained_model = 'models/VGG_imagenet.npy',
        cfg_file = 'cfg/faster_rcnn_end2end.yml'):
        """
        Train a given model with the provided data.
        
        """

        # hyper-parameters
        # ------------
        start_step = 0
        lr_decay = 1./10
        # ------------

        np.random.seed(1024)

        # load config
        cfg_from_file(cfg_file)
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
        network.load_pretrained_npy(net, pretrained_model, encoding='latin1')

        net.cuda()
        net.train()

        params = list(net.parameters())
        optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print('START TRAIN.')

        # training
        train_loss = 0
        #tp, tf, fg, bg = 0., 0., 0, 0
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
                save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}.h5'.format(imdb_name, step))
                network.save_net(save_name, net)
                print(('save model: {}'.format(save_name)))

            if step in lr_decay_steps:
                lr *= lr_decay
                optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

            if re_cnt:
                #tp, tf, fg, bg = 0., 0., 0, 0
                train_loss = 0
                step_cnt = 0
                t.tic()
                re_cnt = False
            
        print('FINISH.')

    def test(self, imdb_name='voc_2007_test', trained_model = 'models/VGGnet_fast_rcnn_iter_70000.h5'):
        pass

    def demo(self, imdb_name='voc_2007_test', trained_model = 'models/VGGnet_fast_rcnn_iter_70000.h5', im_file = 'data/004545.jpg'):
        import cv2

        image = cv2.imread(im_file)

        # load image db for classes
        imdb = get_imdb(imdb_name)

        #detector = FasterRCNN(classes=imdb.classes)
        detector = FasterRCNN()
        network.load_net(trained_model, detector)
        detector.cuda()
        detector.eval()
        print('load model successfully!')

        t = Timer()
        t.tic()

        dets, scores, classes = detector.detect(image, 0.7)
        
        runtime = t.toc()
        print(('total spend: {}s'.format(runtime)))

        im2show = np.copy(image)
        for i, det in enumerate(dets):
            det = tuple(int(x) for x in det)
            cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
            cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
        
        cv2.imwrite('out.jpg', im2show)
