import argparse
import configparser
import datetime
import json
import os
import os.path as osp
import cv2

import numpy as np
import pycocotools.mask as rletools
import torch
import tqdm
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_iou

from mots20_to_mot17_gt import load_mots_gt

SPLITS=['train']

def read_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def save_json(data, path):
    with open(path, 'w') as json_file:
        json.dump(data, json_file)


def get_im_anns_dict(anns):
    im_anns = {}
    for im_ann in anns['images']:
        im_anns[im_ann['id']] = im_ann
    return im_anns


def crop_box(im, bbox):
    x1, y1, w, h = bbox
    x2, y2 = x1+ w, y1+ h
    return im.crop((x1, y1, x2, y2))


def ped_im_from_anno(data_root, anno, im_anns):
    im_path = osp.join(data_root, im_anns[anno['image_id']]['file_name'])
    im = Image.open(im_path)

    if anno['mask'] is not None:
        mask = rletools.decode(anno['mask'].mask)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        im = im * mask
        im = Image.fromarray(im)

    crop_im = crop_box(im, anno['bbox'])
    return crop_im


def get_img_id(dataset, seq, fname):
    return int(f"{dataset[3:5]}{seq.split('-')[1]}{int(fname.split('.')[0]):06}")


def read_seqinfo(path):
    cp = configparser.ConfigParser()
    cp.read(path)

    return {'height': int(cp.get('Sequence', 'imHeight')),
            'width': int(cp.get('Sequence', 'imWidth')),
            'fps': int(cp.get('Sequence', 'frameRate')),
            'seq_length': int(cp.get('Sequence', 'seqLength'))}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--no_bg', action='store_true')
    parser.add_argument('--mask_dir', default='')

    args = parser.parse_args()

    if args.no_bg:
        assert args.mask_dir

    print(f'DATASET: {args.dataset}')
    print(f'NO_BG: {args.no_bg}')

    data_path = osp.join(args.data_root, args.dataset)

    # generate reid data
    reid_imgs_path = osp.join(data_path, 'reid')
    os.makedirs(reid_imgs_path, exist_ok=True)

    for split in SPLITS:
        print(f"Processing sequence {split} in dataset {args.dataset}")

        seq_path = osp.join(data_path, split)
        # seqinfo_path = osp.join(seq_path, 'seqinfo.ini')
        gt_path = osp.join(data_path, 'annotations', f'annotation_{split}.odgt')
        with open(gt_path, 'r+') as anno_file:
            datalist = anno_file.readlines()
        im_dir = seq_path

        images = []
        imgs_list_dir = os.listdir(im_dir)
        for i, img in tqdm.tqdm(enumerate(sorted(imgs_list_dir))):
            im = cv2.imread(os.path.join(im_dir, img))
            h, w, _ = im.shape

            # data['images'].append({
            #     "file_name": img,
            #     "height": h,
            #     "width": w,
            #     "id": i, })

            images.append({
                "file_name": img,
                "height": h,
                "width": w,
                "id": i, })

        annotation_id = 0
        frame_n = 0
        img_file_name_to_id = {
            os.path.splitext(img_dict['file_name'])[0]: img_dict['id']
            for img_dict in images}
            # for img_dict in data['images']}

        annotations = []
        ignores = 0
        for da in tqdm.tqdm(datalist):
            json_data = json.loads(da)
            gtboxes = json_data['gtboxes']
            for gtbox in gtboxes:
                if gtbox['tag'] == 'person':
                    bbox = gtbox['fbox']
                    area = bbox[2] * bbox[3]

                    ignore = False
                    visibility = 1.0
                    # if 'occ' in gtbox['extra']:
                    #     visibility = 1.0 - gtbox['extra']['occ']
                    # if visibility <= VIS_THRESHOLD:
                    #     ignore = True

                    if 'ignore' in gtbox['extra']:
                        ignore = ignore or bool(gtbox['extra']['ignore'])

                    ignores += int(ignore)

                    annotation = {
                        "ped_id": annotation_id,
                        "frame_n": frame_n,
                        "category_id": 1,
                        "id": annotation_id,
                        "image_id": img_file_name_to_id[json_data['ID']],
                        "bbox": bbox,
                        "area": area,
                        "vis": visibility,
                        "iscrowd": 0,
                        "mask": None, }

                    annotation_id += 1
                    frame_n += 1
                    # data['annotations'].append(annotation)
                    annotations.append(annotation)

        for img in tqdm.tqdm(images):
            im_path = osp.join(im_dir, img['file_name'])
            im = Image.open(im_path)

            annos = [anno for anno in annotations if anno['image_id'] == img['id']]
            for anno in annos:
                if anno['mask'] is not None:
                    mask = rletools.decode(anno['mask'].mask)
                    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

                    im = im * mask
                    im = Image.fromarray(im)

                crop_im = crop_box(im, anno['bbox'])
                box_path = osp.join(reid_imgs_path, f"{anno['id']}.png")
                crop_im.save(box_path)

        # save annotation file
        ann_dir = data_path
        if not osp.exists(ann_dir):
            os.makedirs(ann_dir)
        os.makedirs(ann_dir, exist_ok=True)

        # remove mask before saving
        for i in range(len(annotations)):
            annotations[i]['mask'] = None

        for img in images:
            img_name = os.path.splitext(img['file_name'])[0]

            data = {'info': {'sequence': img_name,
                     'dataset': args.dataset,
                     'split': split,
                     'creation_date': datetime.datetime.today().strftime('%Y-%m-%d-%H-%M'),},
                'images': [img],
                'annotations': [anno for anno in annotations if anno['image_id'] == img['id']],
                'categories': [{'id': 1, 'name': 'person', 'supercategory': 'person'}]}

            ann_file = osp.join(ann_dir, f"{img_name}.json")
            if args.no_bg:
                ann_file = osp.join(ann_dir, f"{img_name}_NO_BG.json")

                # remove mask before saving
                # for i in range(len(data['annotations'])):
                #     data['annotations'][i]['mask'] = None

            save_json(data, ann_file)
            print(f"Saving annotation file in {ann_file}.")


if __name__ == '__main__':
    main()
