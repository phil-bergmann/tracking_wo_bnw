import numpy as np
import cv2
import os
import os.path as osp
import configparser
import csv
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from ..config import cfg
from .mot_sequence import MOTSequence


class MOT15Sequence(MOTSequence):
    """Loads a sequence from the 2DMOT15 dataset.

    This dataloader is designed so that it can handle only one sequence, if more have to be handled
    at once one should use a wrapper class.
    """

    def __init__(self, seq_name=None, vis_threshold=0.0, normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225]):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self.vis_threshold = vis_threshold

        self._mot_dir = osp.join(cfg.DATA_DIR, '2DMOT2015')

        self._train_folders = ['Venice-2', 'KITTI-17', 'KITTI-13', 'ADL-Rundle-8', 'ADL-Rundle-6', 'ETH-Pedcross2',
                               'ETH-Sunnyday', 'ETH-Bahnhof', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte']
        self._test_folders = ['Venice-1', 'KITTI-19', 'KITTI-16', 'ADL-Rundle-3', 'ADL-Rundle-1', 'AVG-TownCentre',
                              'ETH-Crossing', 'ETH-Linthescher', 'ETH-Jelmoli', 'PETS09-S2L2', 'TUD-Crossing']

        self.transforms = ToTensor()

        if seq_name:
            assert seq_name in self._train_folders or seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(seq_name)

            self.data, self.no_gt = self.sequence(seq_name)
        else:
            self.data = []
            self.no_gt = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        data = self.data[idx]
        img = Image.open(data['im_path']).convert("RGB")
        img = self.transforms(img)

        sample = {}
        sample['img'] = img
        sample['dets'] = torch.tensor([det[:4] for det in data['dets']])
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']

        return sample

    def __str__(self):
        return self._seq_name

    def sequence(self, seq_name):
        if seq_name in self._train_folders:
            seq_path = osp.join(self._mot_dir, 'train', seq_name)
        else:
            seq_path = osp.join(self._mot_dir, 'test', seq_name)

        im_dir = osp.join(seq_path, 'img1')
        gt_file = osp.join(seq_path, 'gt', 'gt.txt')
        det_file = osp.join(seq_path, 'det', 'det.txt')

        total = []

        boxes = {}
        dets = {}
        visibility = {}

        valid_files = [f for f in os.listdir(im_dir) if len(re.findall("^[0-9]{6}[.][j][p][g]$", f)) == 1]
        seq_length = len(valid_files)

        for i in range(1, seq_length+1):
            boxes[i] = {}
            dets[i] = []
            visibility[i] = {}

        no_gt = False
        if osp.exists(gt_file):
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    #
                    if int(row[6]) == 1: #and float(row[8]) >= self.vis_threshold:
                        # Make pixel indexes 0-based, should already be 0-based (or not)
                        x1 = int(row[2]) - 1
                        y1 = int(row[3]) - 1
                        # This -1 accounts for the width (width of 1 x1=x2)
                        x2 = x1 + float(row[4]) - 1
                        y2 = y1 + float(row[5]) - 1
                        bb = np.array([x1,y1,x2,y2], dtype=np.float32)
                        boxes[int(row[0])][int(row[1])] = bb
                        visibility[int(row[0])][int(row[1])] = float(row[8])
        else:
            no_gt = True

        if osp.exists(det_file):
            with open(det_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    if len(row) > 0:
                        x1 = float(row[2]) - 1
                        y1 = float(row[3]) - 1
                        # This -1 accounts for the width (width of 1 x1=x2)
                        x2 = x1 + float(row[4]) - 1
                        y2 = y1 + float(row[5]) - 1
                        score = float(row[6])
                        bb = np.array([x1,y1,x2,y2, score], dtype=np.float32)
                        dets[int(row[0])].append(bb)

        for i in range(1,seq_length+1):
            im_path = osp.join(im_dir,"{:06d}.jpg".format(i))

            sample = { 'gt':boxes[i],
                       'im_path':im_path,
                       'dets':dets[i],
                       'vis':visibility[i],
            }

            total.append(sample)

        return total, no_gt
