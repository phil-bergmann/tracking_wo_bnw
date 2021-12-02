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


def load_results(res_file):
    results = {}

    assert os.path.isfile(res_file), res_file

    with open(res_file, "r") as of:
        csv_reader = csv.reader(of, delimiter=',')
        for row in csv_reader:
            if int(row[6]) == 1 and int(row[7]) == 1: # and float(row[8]) > 0:
                frame_id = int(row[0])
                track_id = int(row[1])

                if not frame_id in results:
                    # results[frame_id] = {'boxes': [], 'scores': [], 'labels': []}
                    results[frame_id] = {}

                x1 = int(row[2]) - 1
                y1 = int(row[3]) - 1
                w = int(row[4]) - 1
                h = int(row[5]) - 1

                results[frame_id][track_id] = {}
                results[frame_id][track_id]['bbox'] = [x1, y1, w, h]
                results[frame_id][track_id]['score'] = float(row[6])
                results[frame_id][track_id]['label '] = 1
                results[frame_id][track_id]['visibility'] = float(row[8])

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mot17_gt_file', required=True)
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
    for frame_id in mask_objects_per_frame.keys():
        mask_objects_per_frame[frame_id] = [
            obj for obj in mask_objects_per_frame[frame_id]
            if obj.class_id == 2]

        for obj in mask_objects_per_frame[frame_id]:
            obj.track_id = obj.track_id % 1000

    mot17_objects_per_frame = load_results(args.mot17_gt_file)

    track_ids_in_mots20 = list(set([
        obj.track_id for objs in mask_objects_per_frame.values()
        for obj in objs]))
    track_ids_in_mot17 = list(set([obj for objs in mot17_objects_per_frame.values() for obj in objs]))

    cost_iou = torch.zeros(len(track_ids_in_mots20), len(track_ids_in_mot17)).float()

    for frame_id in mask_objects_per_frame.keys():
        for mask_object in mask_objects_per_frame[frame_id]:
            input_track_id = mask_object.track_id
            input_bbox = torch.from_numpy(rletools.toBbox(mask_object.mask))[None, ...]
            # x1y1wh to x1y1x2y2
            input_bbox[:, 2:] += input_bbox[:, :2]

            for mot17_track_id, mot17_object in mot17_objects_per_frame[frame_id].items():
                mot17_bbox = torch.tensor(mot17_object['bbox'])[None, ...]
                # x1y1wh to x1y1x2y2
                mot17_bbox[:, 2:] += mot17_bbox[:, :2]

                if torch.isnan(box_iou(input_bbox, mot17_bbox)):
                    print(input_bbox, mot17_bbox)
                    print(box_iou(input_bbox, mot17_bbox))
                    exit()

                cost_iou[
                    track_ids_in_mots20.index(input_track_id),
                    track_ids_in_mot17.index(mot17_track_id)] += box_iou(input_bbox, mot17_bbox)[0].item()

    cost_iou = 1.0 / (cost_iou + 1e-8)
    mots20_inds, mot17_inds = linear_sum_assignment(cost_iou)
    mots20_inds = mots20_inds.tolist()
    mot17_inds = mot17_inds.tolist()

    for frame_id, mask_objects in mask_objects_per_frame.items():

        for mask_object in mask_objects_per_frame[frame_id]:
            mot17_track_id = track_ids_in_mot17[mot17_inds[mots20_inds.index(track_ids_in_mots20.index(mask_object.track_id))]]

            mot17_objects = mot17_objects_per_frame[frame_id]

            if mot17_track_id in mot17_objects:
                x1, y1, w, h = [int(b) for b in mot17_objects[mot17_track_id]['bbox']]
                vis = mot17_objects[mot17_track_id]['visibility']
            else:
                x1, y1, w, h = [int(b) for b in rletools.toBbox(mask_object.mask)]
                vis = 1.0

            bounding_boxes.append(
                [frame_id, mask_object.track_id, x1, y1, w, h, 1, 1, vis]
            )

    with open(args.output_gt_file, "w") as of:
        writer = csv.writer(of, delimiter=',')
        for bbox in bounding_boxes:
            writer.writerow(bbox)
