# --------------------------------------------------------
# Pytorch FPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang, based on code from faster R-CNN
# --------------------------------------------------------

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import pdb
import pprint
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

import cv2
from fpn.model.fpn.resnet import FPNResNet
from fpn.model.utils.config import (cfg, cfg_from_file, cfg_from_list,
                                    get_output_dir)
from fpn.model.utils.net_utils import (adjust_learning_rate, clip_gradient,
                                       load_net, save_checkpoint, save_net,
                                       vis_detections, weights_normal_init)
from fpn.model.utils.summary import *
from fpn.roi_data_layer.roibatchLoader import roibatchLoader
from fpn.roi_data_layer.roidb import combined_roidb
from fpn.test import validate
from tensorboardX import SummaryWriter


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network with FPN')
    parser.add_argument('exp_name', type=str, help='experiment name')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='res101, res152, etc',
                        default='res101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=500, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="output/fpn", )
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=4, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--lscale', dest='lscale',
                        help='whether use large scale',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # pretrained trained model
    parser.add_argument('--pre_checkpoint', dest='pre_checkpoint',
                        help='path to pretrained file',
                        default=None, type=str)
    parser.add_argument('--pre_file', dest='pre_file',
                        help='path to pretrained config',
                        default=None, type=str)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        action='store_true')
    parser.add_argument('--resume_exp_name', dest='resume_exp_name',
                        help='exp_name to load model', type=str)
    parser.add_argument('--resume_session', dest='resume_session',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--resume_epoch', dest='resume_epoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    # log and diaplay
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        help='whether use tensorflow tensorboard',
                        default=False, type=bool)

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        num_data = train_size
        self.num_per_batch = int(num_data / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if num_data % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, num_data).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        # rand_num = torch.arange(self.num_per_batch).long().view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


def _print(str, logger=None):
    print(str)
    if logger is None:
        return
    logger.info(str)


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.use_tfboard:
        writer = SummaryWriter(os.path.join(args.save_dir, 'runs', args.net, args.dataset, args.exp_name))

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_0712_trainval"
        args.imdbval_name = "voc_0712_test"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "mot_2017_train":
        args.imdb_name = "mot_2017_train"
        args.imdbval_name = "mot_2017_test"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '100']
    elif args.dataset == "mot_2017_small_train":
        args.imdb_name = "mot_2017_small_train"
        args.imdbval_name = "mot_2017_smallval"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '100']
    elif args.dataset == "mot_2017_seq_1":
        args.imdb_name = "mot_2017_seq_train_1"
        args.imdbval_name = "mot_2017_seq_val_1"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '100']
    elif args.dataset == "mot_2017_seq_2":
        args.imdb_name = "mot_2017_seq_train_2"
        args.imdbval_name = "mot_2017_seq_val_2"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '100']
    elif args.dataset == "mot_2017_seq_3":
        args.imdb_name = "mot_2017_seq_train_3"
        args.imdbval_name = "mot_2017_seq_val_3"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '100']
    elif args.dataset == "mot_2017_seq_4":
        args.imdb_name = "mot_2017_seq_train_4"
        args.imdbval_name = "mot_2017_seq_val_4"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '100']
    elif args.dataset == "mot_2017_seq_5":
        args.imdb_name = "mot_2017_seq_train_5"
        args.imdbval_name = "mot_2017_seq_val_5"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '100']
    elif args.dataset == "mot_2017_seq_6":
        args.imdb_name = "mot_2017_seq_train_6"
        args.imdbval_name = "mot_2017_seq_val_6"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '100']
    elif args.dataset == "mot_2017_seq_7":
        args.imdb_name = "mot_2017_seq_train_7"
        args.imdbval_name = "mot_2017_seq_val_7"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '100']
    elif args.dataset == "mot19_cvpr_train":
        args.imdb_name = "mot19_cvpr_train"
        args.imdbval_name = "mot19_cvpr_test"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '100']
    elif args.dataset == "mot19_cvpr_seq_1":
        args.imdb_name = "mot19_cvpr_seq_train_1"
        args.imdbval_name = "mot19_cvpr_seq_val_1"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '100']
    elif args.dataset == "mot19_cvpr_seq_2":
        args.imdb_name = "mot19_cvpr_seq_train_2"
        args.imdbval_name = "mot19_cvpr_seq_val_2"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '100']
    elif args.dataset == "mot19_cvpr_seq_3":
        args.imdb_name = "mot19_cvpr_seq_train_3"
        args.imdbval_name = "mot19_cvpr_seq_val_3"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '100']
    elif args.dataset == "mot19_cvpr_seq_4":
        args.imdb_name = "mot19_cvpr_seq_train_4"
        args.imdbval_name = "mot19_cvpr_seq_val_4"
        set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                    'MAX_NUM_GT_BOXES', '100']
    else:
        raise NotImplementedError

    # load config from pre file
    if args.pre_file is not None:
        cfg_file = args.pre_file
        cfg_from_file(cfg_file)

    # load changes from current config file
    cfg_file = f"src/fpn/cfgs/{args.net}{'_ls' if args.lscale else ''}.yml"
    cfg_from_file(cfg_file)

    # load changes from set_cfg list
    cfg_from_list(set_cfgs)
    cfg.CUDA = args.cuda

    print('Using config:')
    pprint.pprint(cfg)

    # set seeds and make deterministic
    torch.backends.cudnn.fastest = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)

    output_dir = os.path.join(args.save_dir, args.net,
                              args.dataset, args.exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # data
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    # train
    cfg.USE_GPU_NMS = args.cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    _print('[TRAIN] {:d} roidb entries'.format(len(roidb)))
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size,
                             imdb.num_classes, training=True)
    sampler_batch = sampler(train_size, args.batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers)

    # evaluation
    imdb_train, roidb_train, ratio_list_train, ratio_index_train = combined_roidb(args.imdb_name, False)
    imdb_train.competition_mode(on=True)
    dataset_train = roibatchLoader(roidb_train, ratio_list_train, ratio_index_train, 1,
                                   imdb_train.num_classes, training=False, normalize=False)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1,
                                                   shuffle=False, num_workers=args.num_workers,
                                                   pin_memory=True)

    imdb_val, roidb_val, ratio_list_val, ratio_index_val = combined_roidb(args.imdbval_name, False)
    imdb_val.competition_mode(on=True)
    _print('[VAL] {:d} roidb entries'.format(len(roidb_val)))
    dataset_val = roibatchLoader(roidb_val, ratio_list_val, ratio_index_val, 1,
                                 imdb_val.num_classes, training=False, normalize=False)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
                                                 shuffle=False, num_workers=args.num_workers,
                                                 pin_memory=True)

    # initilize the network here.
    if args.net == 'res101':
        FPN = FPNResNet(imdb.classes, 101, pretrained=True)
    elif args.net == 'res50':
        FPN = FPNResNet(imdb.classes, 50, pretrained=True)
    elif args.net == 'res152':
        FPN = FPNResNet(imdb.classes, 152, pretrained=True)
    else:
        print("Network is not defined.")
        pdb.set_trace()

    FPN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    params = []
    for key, value in dict(FPN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    # resume or pretrained
    if args.resume and args.pre_checkpoint is not None:
        raise NotImplementedError

    if args.pre_checkpoint is not None:
        checkpoint = torch.load(args.pre_checkpoint)
        model_state_dict = FPN.state_dict()
        state_dict = {k: v
                    for k, v in checkpoint['model'].items()
                    if v.shape == model_state_dict[k].shape}
        model_state_dict.update(state_dict)
        FPN.load_state_dict(model_state_dict)

    if args.resume:
        load_name = os.path.join(os.path.join(args.save_dir, args.net, args.dataset, args.resume_exp_name),
                                 f'fpn_{args.resume_session}_{args.resume_epoch}.pth')
        _print("loading checkpoint %s" % (load_name), )
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch'] + 1
        FPN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']

        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

        _print("loaded checkpoint %s" % (load_name), )

    # initilize the tensor holder here.
    im_data = Variable(torch.FloatTensor(1))
    im_info = Variable(torch.FloatTensor(1))
    num_boxes = Variable(torch.LongTensor(1))
    gt_boxes = Variable(torch.FloatTensor(1))

    # ship to cuda
    if args.cuda:
        # torch.backends.cudnn.benchmark = True
        FPN.cuda()
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
    elif torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if args.mGPUs:
        FPN = nn.DataParallel(FPN)

    # training
    iters_per_epoch = int(train_size / args.batch_size)
    for epoch in range(args.start_epoch, args.max_epochs):
        # setting to train mode
        FPN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)

        for step in range(iters_per_epoch):
            data = data_iter.next()
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            FPN.zero_grad()
            # try:
            _, _, _, rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            roi_labels = FPN(im_data, im_info, gt_boxes, num_boxes)
            # except:
            #     print(data[4], gt_boxes, num_boxes)
            #     img = (data[0].permute(0, 2, 3, 1)[0].numpy() + cfg.PIXEL_MEANS).astype(np.uint8).copy()
            #
            #     bbox = data[2][0].cpu().numpy()
            #     im2show = vis_detections(img, 'anything', bbox, 0.0)
            #     print(bbox)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.data[0]

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().data[0]
                    loss_rpn_box = rpn_loss_box.mean().data[0]
                    loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
                    loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
                    fg_cnt = torch.sum(roi_labels.data.ne(0))
                    bg_cnt = roi_labels.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.data[0]
                    loss_rpn_box = rpn_loss_box.data[0]
                    loss_rcnn_cls = RCNN_loss_cls.data[0]
                    loss_rcnn_box = RCNN_loss_bbox.data[0]
                    fg_cnt = torch.sum(roi_labels.data.ne(0))
                    bg_cnt = roi_labels.data.numel() - fg_cnt

                _print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                       % (args.session, epoch, step, iters_per_epoch, loss_temp, lr), )
                _print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start), )
                _print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                       % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box), )

                if args.use_tfboard:
                    scalars = [loss_temp, loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box]
                    names = ['loss', 'loss_rpn_cls', 'loss_rpn_box', 'loss_rcnn_cls', 'loss_rcnn_box']
                    write_scalars(writer, scalars, names, iters_per_epoch * (epoch - 1) + step, tag='train_loss')

                loss_temp = 0
                start = time.time()

        save_name = os.path.join(output_dir, f'fpn_{args.session}_{epoch}.pth')
        save_checkpoint({
            'session': args.session,
            'epoch': epoch,
            'model': FPN.module.state_dict() if args.mGPUs else FPN.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, save_name)
        _print('[Save]: {}'.format(save_name), )

        with open(os.path.join(output_dir, 'config.yaml'), 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)

        end = time.time()
        # print(end - start)

        #
        # evaluation
        #

        #
        # train
        #
        all_boxes = validate(FPN, dataloader_train, imdb_train,
                             vis=False, cuda=args.cuda)

        # evaluate without print output
        sys.stdout = open(os.devnull, 'w')
        aps = imdb.evaluate_detections(all_boxes, output_dir)
        sys.stdout = sys.__stdout__

        # print because of flushing in imdd_eval.evaluate_detections
        _print("")
        _print(f'[TRAIN]: Mean AP = {np.mean(aps):.4f}')
        if args.use_tfboard:
            write_scalars(writer, [np.mean(aps)], ['train'], epoch, tag='mean_ap')

        #
        # val
        #
        all_boxes = validate(FPN, dataloader_val, imdb_val,
                             vis=False, cuda=args.cuda)

        # evaluate without print output
        sys.stdout = open(os.devnull, 'w')
        aps = imdb_val.evaluate_detections(all_boxes, output_dir)
        sys.stdout = sys.__stdout__

        # print because of flushing in imdd_eval.evaluate_detections
        _print("")
        _print(f'[VAL]: Mean AP = {np.mean(aps):.4f}')
        if args.use_tfboard:
            write_scalars(writer, [np.mean(aps)], ['val'], epoch, tag='mean_ap')
