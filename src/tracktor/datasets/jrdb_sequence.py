import configparser
import csv
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import cv2

from ..config import cfg
from torchvision.transforms import ToTensor
from functools import cmp_to_key

class JRDB_Sequence(Dataset):
    """JackRabbot Dataset.

    This dataloader is designed so that it can handle only one sequence, if more have to be
    handled one should inherit from this class.
    """

    def __init__(self, seq_name=None, dets='', vis_threshold=0.0, det_threshold=0.0,
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self._dets = dets
        self._vis_threshold = vis_threshold

        # Filter public detections with confidence threshold
        self._det_threshold = det_threshold

        self._mot_dir = osp.join(cfg.DATA_DIR, 'JRDB')

        self._train_folders = os.listdir(os.path.join(self._mot_dir, 'sequences'))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, 'test_sequences'))

        self.transforms = ToTensor()

        assert seq_name in self._train_folders or seq_name in self._test_folders, \
            'Image set does not exist: {}'.format(seq_name)

        self.data, self.no_gt = self._sequence()

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

    def _sequence(self):
        seq_name = self._seq_name
        if seq_name in self._train_folders:
            seq_path = osp.join(self._mot_dir, 'sequences', seq_name)
        else:
            seq_path = osp.join(self._mot_dir, 'test_sequences', seq_name)

        imDir = osp.join(seq_path, 'imgs')
        seqLength = len(os.listdir(imDir))
        gt_file = osp.join(seq_path, 'gt', 'gt.txt')

        total = []
        train = []
        val = []

        visibility = {}
        boxes = {}
        dets = {}

        for i in range(0, seqLength):
            boxes[i] = {}
            visibility[i] = {}
            dets[i] = []

        no_gt = False
        if osp.exists(gt_file):
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    # Pixels are already 0-based
                    x1 = int(row[2])
                    y1 = int(row[3])
                    x2 = x1 + int(row[4])
                    y2 = y1 + int(row[5])
                    bb = np.array([x1,y1,x2,y2], dtype=np.float32)
                    boxes[int(row[0])][int(row[1])] = bb
        else:
            no_gt = True

        det_file = osp.join(seq_path, 'det', 'det.txt')

        if osp.exists(det_file):
            with open(det_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    x1 = float(row[2])
                    y1 = float(row[3])
                    x2 = x1 + float(row[4])
                    y2 = y1 + float(row[5])
                    score = float(row[6])
                    if score > self._det_threshold:
                        bb = np.array([x1,y1,x2,y2, score], dtype=np.float32)
                        dets[int(row[0])].append(bb)

        for i in range(seqLength):
            im_path = osp.join(imDir,"{:06d}.jpg".format(i))
            sample = {'gt':boxes[i],
                      'im_path':im_path,
                      'vis':visibility[i],
                      'dets':dets[i],}

            total.append(sample)

        return total, no_gt

    def __str__(self):
        return f"{self._seq_name}-{self._dets[:-2]}"

    def write_results(self, all_tracks, output_dir):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, [7 3D parameters], <conf>

        Instructions to sumbit:
        There should be 27 sequences in test submission. Zip such that the txt files are directly below zip root.
        """

        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file = osp.join(output_dir, self._seq_name + '_image_stitched.txt')

        results = []
        for i, track in all_tracks.items():
            for frame, bb in track.items():
                x1 = bb[0]
                y1 = bb[1]
                x2 = bb[2]
                y2 = bb[3]
                # frame, track_id, x, y, w, h, [7 3D parameters], [optional confidence]
                results.append([frame, i, x1, y1, x2-x1, y2-y1, -1, -1, -1, -1, -1, -1, -1, 1])

        results.sort(key=cmp_to_key(lambda x, y: x[0] - y[0]))
        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for row in results:
                writer.writerow(row)
