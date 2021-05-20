import sys
import os
# sys.path.append('/usr/stud/brasoand/motsyn_iccv21')
import json
import os.path as osp
from PIL import Image
import tqdm
import glob

#ANN_PATH = '/storage/remote/atcremers82/mot_neural_solver/sanity_check_data/comb_annotations/split_mot17.json'  # JSON annotations for dataset
ANN_PATHS = glob.glob('data/MOT17_with_MOTS20_GT_Anonymous_BodyCIAV5_MOTSyn/train/MOT17*.json')
REID_IMAGES_PATH = 'data/MOT17_with_MOTS20_GT_Anonymous_BodyCIAV5_MOTSyn/train/reid' # Path where you want to store ReID images (i.e. one image per detection / gt box)
DATASET_PATH = 'data' # MOTChallenge ROOT (i.e. path where you have a diff directory for MOT20, MOT17, etc.)


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


def ped_im_from_anno(anno, im_anns):
    im_path = osp.join(DATASET_PATH, im_anns[anno['image_id']]['file_name'])
    im = Image.open(im_path)
    crop_im = crop_box(im, anno['bbox'])
    return crop_im


def main():
    os.makedirs(REID_IMAGES_PATH, exist_ok=True)

    for ann_path in ANN_PATHS:
        print(ann_path)
        anns = read_json(ann_path)
        im_anns = get_im_anns_dict(anns)

        for anno in tqdm.tqdm(anns['annotations']):
            box_im = ped_im_from_anno(anno, im_anns)
            box_path = osp.join(REID_IMAGES_PATH, f"{anno['id']}.png")
            box_im.save(box_path)


if __name__ == '__main__':
    main()
