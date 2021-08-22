import argparse
import csv
import os

import numpy as np
import pycocotools.mask as rletools
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_iou
from tracktor.datasets.mot_sequence import MOTSequence


class SegmentedObject:
    """
    Helper class for segmentation objects.
    """
    def __init__(self, mask: dict, class_id: int, track_id: int) -> None:
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id


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
            if frame not in track_ids_per_frame:
                track_ids_per_frame[frame] = set()
            if int(fields[1]) in track_ids_per_frame[frame]:
                assert False, f"Multiple objects with track id {fields[1]} in frame {fields[0]}"
            else:
                track_ids_per_frame[frame].add(int(fields[1]))

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
            objects_per_frame[frame].append(SegmentedObject(
                mask,
                class_id,
                int(fields[1])
            ))

    return objects_per_frame



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mot17_seq', required=True)
    parser.add_argument('--mots20_gt_file', required=True)
    parser.add_argument('--output_gt_file', required=True)

    args = parser.parse_args()

    assert os.path.exists(args.mots20_gt_file), \
        f'MOTS20 GT file does not exist: {args.mots20_gt_file}'

    output_dir = os.path.dirname(args.output_gt_file)
    assert os.path.isdir(output_dir) or not output_dir, \
        f'Directory for output file does not exist: {output_dir}'


    bounding_boxes = []
    mask_objects_per_frame = load_mots_gt(args.mots20_gt_file)

    dataset = MOTSequence(args.mot17_seq, 'data/MOT17')

    for frame_id, mask_objects in mask_objects_per_frame.items():
        frame_data = dataset.data[frame_id - 1]
        mask_objects = [o for o in mask_objects if o.class_id == 2]

        mot17_ids = [track_id for track_id in frame_data['gt'].keys()]
        mot17_boxes = torch.from_numpy(np.stack([frame_data['gt'][i] for i in mot17_ids])).float()
        mots20_boxes = torch.from_numpy(np.stack([
            rletools.toBbox(mask_object.mask)
            for mask_object in mask_objects])).float()
        # x1y1wh to x1y1x2y2
        mots20_boxes[:, 2:] += mots20_boxes[:, :2]
        iou = box_iou(mots20_boxes, mot17_boxes)
        _, col_ind = linear_sum_assignment(1.0 / (iou + 1e-8))

        for ind in col_ind:

            x1, y1, x2, y2 = frame_data['gt'][mot17_ids[ind]].astype(int)

            bounding_boxes.append(
                [frame_id, mot17_ids[ind], x1, y1, x2 - x1, y2 - y1, 1, 1, frame_data['vis'][mot17_ids[ind]]]
            )

    with open(args.output_gt_file, "w") as of:
        writer = csv.writer(of, delimiter=',')
        for bbox in bounding_boxes:
            writer.writerow(bbox)
