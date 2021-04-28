import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, ToTensor)

from ..config import get_output_dir
from .mot_sequence import MOTSequence


class MOTreID(MOTSequence):
    """Multiple Object Tracking Dataset.

    This class builds samples for training of a simaese net. It returns a tripple of 2 matching and 1 not
    matching image crop of a person. The crops con be precalculated.

    Values for P are normally 18 and K 4.
    """

    def __init__(self, seq_name, mot_dir, vis_threshold, P, K, max_per_person, crop_H, crop_W,
                transform, random_triplets=True, normalize_mean=None, normalize_std=None, logger=print):
        super().__init__(seq_name, mot_dir, vis_threshold=vis_threshold)

        self.P = P
        self.K = K
        self.max_per_person = max_per_person
        self.crop_H = crop_H
        self.crop_W = crop_W
        self.logger = logger
        self.random_triplets = random_triplets

        if transform == "random":
            self.transform = Compose([
                RandomCrop((crop_H, crop_W)),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(normalize_mean, normalize_std)])
        elif transform == "center":
            self.transform = Compose([
                CenterCrop((crop_H, crop_W)),
                ToTensor(),
                Normalize(normalize_mean, normalize_std)])
        else:
            raise NotImplementedError(f"Transformation not understood: {transform}")

        self.data = self.build_samples()

    def __getitem__(self, idx):
        """Return the ith triplet"""

        res = []
        # idx belongs to the positive sampled person
        pos = self.data[idx]
        if self.random_triplets:
            res.append(pos[np.random.choice(pos.shape[0], self.K, replace=False)])

            # exclude idx here
            neg_indices = np.random.choice([
                i for i, _ in enumerate(self.data)
                if i != idx], self.P-1, replace=False)
            for i in neg_indices:
                neg = self.data[i]
                res.append(neg[np.random.choice(neg.shape[0], self.K, replace=False)])
        else:
            res.append(pos[np.linspace(0, pos.shape[0] - 1, num=self.K, dtype=int)])

            # exclude idx here
            neg_indices = [i for i, _ in enumerate(self.data) if i != idx]
            neg_indices_choice = np.linspace(0, len(self.data) - 2, num=self.P-1, dtype=int)
            neg_indices = np.array(neg_indices)[neg_indices_choice].tolist()

            for i in neg_indices:
                neg = self.data[i]
                res.append(neg[np.linspace(0, neg.shape[0] - 1, num=self.K, dtype=int)])

        # concatenate the results
        r = []
        for pers in res:
            for im in pers:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(im)
                r.append(self.transform(im))
        images = torch.stack(r, 0)

        # construct the labels
        labels = [idx] * self.K

        for l in neg_indices:
            labels += [l] * self.K

        labels = np.array(labels)

        batch = [images, labels]

        return batch

    def build_samples(self):
        """Builds the samples out of the sequence."""

        tracks = {}

        for sample in self.data:
            for k, v in sample['gt'].items():
                track = {'id': k, 'im_path': sample['im_path'], 'gt': v}
                if k not in tracks:
                    tracks[k] = []
                tracks[k].append(track)

        # sample maximal self.max_per_person images per person and
        # filter out tracks smaller than self.K samples
        res = []
        for k,v in tracks.items():
            l = len(v)
            if l >= self.K:
                pers = []
                if self.max_per_person is not None and l > self.max_per_person:
                    for i in np.random.choice(l, self.max_per_person, replace=False):
                        pers.append(self.build_crop(v[i]['im_path'], v[i]['gt']))
                else:
                    for i in range(l):
                        pers.append(self.build_crop(v[i]['im_path'], v[i]['gt']))

                res.append(np.array(pers))

        if self._seq_name:
            self.logger(f"[*] Loaded {len(res)} persons from sequence {self._seq_name}.")

        return res

    def build_crop(self, im_path, gt):
        im = cv2.imread(im_path)
        height, width, _ = im.shape
        #blobs, im_scales = _get_blobs(im)
        #im = blobs['data'][0]
        #gt = gt * im_scales[0]
        # clip to image boundary
        w = gt[2] - gt[0]
        h = gt[3] - gt[1]
        context = 0
        gt[0] = np.clip(gt[0]-context*w, 0, width-1)
        gt[1] = np.clip(gt[1]-context*h, 0, height-1)
        gt[2] = np.clip(gt[2]+context*w, 0, width-1)
        gt[3] = np.clip(gt[3]+context*h, 0, height-1)

        im = im[int(gt[1]):int(gt[3]), int(gt[0]):int(gt[2])]

        im = cv2.resize(im, (int(self.crop_W*1.125), int(self.crop_H*1.125)), interpolation=cv2.INTER_LINEAR)

        return im
