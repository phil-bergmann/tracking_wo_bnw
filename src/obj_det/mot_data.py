import configparser
import csv
import os
import os.path as osp
import pickle

import numpy as np
import pycocotools.mask as rletools
import scipy
import torch
from PIL import Image


class MOTObjDetect(torch.utils.data.Dataset):
    """ Data class for the Multiple Object Tracking Dataset
    """

    def __init__(self, root, transforms=None, vis_threshold=0.25,
                 split_seqs=None, frame_range_start=0.0, frame_range_end=1.0):
        self.root = root
        self.transforms = transforms
        self._vis_threshold = vis_threshold
        self._classes = ('background', 'pedestrian')
        self._img_paths = []
        self._split_seqs = split_seqs

        self.mots_gts = {}
        for f in sorted(os.listdir(root)):
            path = os.path.join(root, f)

            if not os.path.isdir(path):
                continue

            if split_seqs is not None and f not in split_seqs:
                continue

            config_file = os.path.join(path, 'seqinfo.ini')

            assert os.path.exists(config_file), \
                'Path does not exist: {}'.format(config_file)

            config = configparser.ConfigParser()
            config.read(config_file)
            seq_len = int(config['Sequence']['seqLength'])
            im_ext = config['Sequence']['imExt']
            im_dir = config['Sequence']['imDir']

            img_dir = os.path.join(path, im_dir)

            start_frame = int(frame_range_start * seq_len)
            end_frame = int(frame_range_end * seq_len)

            # for i in range(seq_len):
            for i in range(start_frame, end_frame):
                img_path = os.path.join(img_dir, f"{i + 1:06d}{im_ext}")
                assert os.path.exists(img_path), f'Path does not exist: {img_path}'
                self._img_paths.append(img_path)

            # print(len(self._img_paths))

            if self.has_masks:
                gt_file = os.path.join(os.path.dirname(img_dir), 'gt', 'gt.txt')
                self.mots_gts[gt_file] = load_mots_gt(gt_file)

    def __str__(self):
        if self._split_seqs is None:
            return self.root
        return f"{self.root}/{self._split_seqs}"

    @property
    def num_classes(self):
        return len(self._classes)

    def _get_annotation(self, idx):
        """
        """

        if 'test' in self.root:
            num_objs = 0
            boxes = torch.zeros((num_objs, 4), dtype=torch.float32)

            return {'boxes': boxes,
                'labels': torch.ones((num_objs,), dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': torch.zeros((num_objs,), dtype=torch.int64),
                'visibilities': torch.zeros((num_objs), dtype=torch.float32)}

        img_path = self._img_paths[idx]
        file_index = int(os.path.basename(img_path).split('.')[0])

        gt_file = os.path.join(os.path.dirname(
            os.path.dirname(img_path)), 'gt', 'gt.txt')

        assert os.path.exists(gt_file), \
            'GT file does not exist: {}'.format(gt_file)

        bounding_boxes = []

        if self.has_masks:
            mask_objects_per_frame = self.mots_gts[gt_file][file_index]
            masks = []
            for mask_object in mask_objects_per_frame:
                # class_id = 1 is car
                # class_id = 2 is pedestrian
                # class_id = 10 IGNORE
                if mask_object.class_id in [1, 10] or not rletools.area(mask_object.mask):
                    continue

                bbox = rletools.toBbox(mask_object.mask)
                x1, y1, w, h = [int(c) for c in bbox]

                bb = {}
                bb['bb_left'] = x1
                bb['bb_top'] = y1
                bb['bb_width'] = w
                bb['bb_height'] = h

                # print(bb, rletools.area(mask_object.mask))

                bb['visibility'] = 1.0
                bb['track_id'] = mask_object.track_id

                masks.append(rletools.decode(mask_object.mask))
                bounding_boxes.append(bb)
        else:
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    visibility = float(row[8])

                    if int(row[0]) == file_index and int(row[6]) == 1 and int(row[7]) == 1 and visibility and visibility >= self._vis_threshold:
                        bb = {}
                        bb['bb_left'] = int(row[2])
                        bb['bb_top'] = int(row[3])
                        bb['bb_width'] = int(row[4])
                        bb['bb_height'] = int(row[5])
                        bb['visibility'] = float(row[8])
                        bb['track_id'] = int(row[1])

                        bounding_boxes.append(bb)

        num_objs = len(bounding_boxes)

        boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
        visibilities = torch.zeros((num_objs), dtype=torch.float32)
        track_ids = torch.zeros((num_objs), dtype=torch.long)

        for i, bb in enumerate(bounding_boxes):
            # Make pixel indexes 0-based, should already be 0-based (or not)
            x1 = bb['bb_left']# - 1
            y1 = bb['bb_top']# - 1
            # This -1 accounts for the width (width of 1 x1=x2)
            x2 = x1 + bb['bb_width']# - 1
            y2 = y1 + bb['bb_height']# - 1

            boxes[i, 0] = x1
            boxes[i, 1] = y1
            boxes[i, 2] = x2
            boxes[i, 3] = y2
            visibilities[i] = bb['visibility']
            track_ids[i] = bb['track_id']

        annos = {'boxes': boxes,
                 'labels': torch.ones((num_objs,), dtype=torch.int64),
                 'image_id': torch.tensor([idx]),
                 'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                 'iscrowd': torch.zeros((num_objs,), dtype=torch.int64),
                 'visibilities': visibilities,
                 'track_ids': track_ids,}

        if self.has_masks:
            # annos['masks'] = torch.tensor(masks, dtype=torch.uint8)
            annos['masks'] = torch.from_numpy(np.stack(masks))
        return annos

    @property
    def has_masks(self):
        return '/MOTS20/' in self.root

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self._img_paths[idx]
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        target = self._get_annotation(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self._img_paths)

    def write_results_files(self, results, output_dir):
        """Write the detections in the format for MOT17Det sumbission

        all_boxes[image] = N x 5 array of detections in (x1, y1, x2, y2, score)

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        Files to sumbit:
        ./MOT17-01.txt
        ./MOT17-02.txt
        ./MOT17-03.txt
        ./MOT17-04.txt
        ./MOT17-05.txt
        ./MOT17-06.txt
        ./MOT17-07.txt
        ./MOT17-08.txt
        ./MOT17-09.txt
        ./MOT17-10.txt
        ./MOT17-11.txt
        ./MOT17-12.txt
        ./MOT17-13.txt
        ./MOT17-14.txt
        """

        #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        files = {}
        for image_id, res in results.items():
            path = self._img_paths[image_id]
            img1, name = osp.split(path)
            # get image number out of name
            frame = int(name.split('.')[0])
            # smth like /train/MOT17-09-FRCNN or /train/MOT17-09
            tmp = osp.dirname(img1)
            # get the folder name of the sequence and split it
            tmp = osp.basename(tmp).split('-')
            # Now get the output name of the file
            out = tmp[0]+'-'+tmp[1]+'.txt'
            outfile = osp.join(output_dir, out)

            # check if out in keys and create empty list if not
            if outfile not in files.keys():
                files[outfile] = []

            if 'masks' in res:
                delimiter = ' '
                # print(torch.unique(res['masks'][0]))
                masks = res['masks'].squeeze(dim=1)# > 0.5 #res['masks'].bool()

                index_map = torch.arange(masks.size(0))[:, None, None]
                index_map = index_map.expand_as(masks)

                masks = torch.logical_and(
                    # remove background
                    masks > 0.5,
                    # remove overlapp by largest probablity
                    index_map == masks.argmax(dim=0)
                )
                for res_i in range(len(masks)):
                    track_id = -1
                    if 'track_ids' in res:
                        track_id = res['track_ids'][res_i].item()
                    mask = masks[res_i]
                    mask = np.asfortranarray(mask)

                    rle_mask = rletools.encode(mask)

                    files[outfile].append(
                        [frame,
                         track_id,
                         2,  # class pedestrian
                         mask.shape[0],
                         mask.shape[1],
                         rle_mask['counts'].decode(encoding='UTF-8')])
            else:
                delimiter = ','
                for res_i in range(len(res['boxes'])):
                    track_id = -1
                    if 'track_ids' in res:
                        track_id = res['track_ids'][res_i].item()
                    box = res['boxes'][res_i]
                    score = res['scores'][res_i]

                    x1 = box[0].item()
                    y1 = box[1].item()
                    x2 = box[2].item()
                    y2 = box[3].item()

                    out = [frame, track_id, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1]

                    if 'keypoints' in res:
                        out.extend(res['keypoints'][res_i][:, :2].flatten().tolist())
                        out.extend(res['keypoints_scores'][res_i].flatten().tolist())

                    files[outfile].append(out)

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=delimiter)
                for d in v:
                    writer.writerow(d)


class SegmentedObject:
    """
    Helper class for segmentation objects.
    """
    def __init__(self, mask: dict, class_id: int, track_id: int, full_bbox=None) -> None:
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id
        self.full_bbox = full_bbox


def load_mots_gt(path: str) -> dict:
    """Load MOTS ground truth from path."""
    objects_per_frame = {}
    track_ids_per_frame = {}  # Check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # Check that no frame contains overlapping masks

    with open(path, "r") as gt_file:
        for line in gt_file:
            line = line.strip()
            fields = line.split(" ")

            frame = int(fields[0])
            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            # if frame not in track_ids_per_frame:
            #     track_ids_per_frame[frame] = set()
            # if int(fields[1]) in track_ids_per_frame[frame]:
            #     assert False, f"Multiple objects with track id {fields[1]} in frame {fields[0]}"
            # else:
            #     track_ids_per_frame[frame].add(int(fields[1]))

            class_id = int(fields[2])
            if not(class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]

            mask = {
                'size': [int(fields[3]), int(fields[4])],
                'counts': fields[5].encode(encoding='UTF-8')}
            if frame not in combined_mask_per_frame:
                combined_mask_per_frame[frame] = mask
            elif rletools.area(rletools.merge([
                    combined_mask_per_frame[frame], mask],
                    intersect=True)):
                assert False, "Objects with overlapping masks in frame " + fields[0]
            else:
                combined_mask_per_frame[frame] = rletools.merge(
                    [combined_mask_per_frame[frame], mask],
                    intersect=False)

            full_bbox = None
            if len(fields) == 10:
                full_bbox = [int(fields[6]), int(fields[7]), int(fields[8]), int(fields[9])]

            objects_per_frame[frame].append(SegmentedObject(
                mask,
                class_id,
                int(fields[1]),
                full_bbox
            ))

    return objects_per_frame
