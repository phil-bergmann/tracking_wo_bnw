import os
import os.path as osp
import numpy as np
import json

import configparser
import datetime

MOT_DATA_ROOT = 'data'
DATASETS = ['MOT17_with_MOTS20_GT_Anonymous_BodyCIAV5_MOTSyn_STEP1']
SPLITS=['train']
ANN_DIR = 'annotations'

def get_img_id(dataset, seq, fname):
    return int(f"{dataset[3:5]}{seq.split('-')[-1]}{int(fname.split('.')[0]):06}")

def read_seqinfo(path):
    cp = configparser.ConfigParser()
    cp.read(path)

    return {'height': int(cp.get('Sequence', 'imHeight')),
            'width': int(cp.get('Sequence', 'imWidth')),
            'fps': int(cp.get('Sequence', 'frameRate'))}

def read_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data

def save_json(data, path):
    with open(path, 'w') as json_file:
        json.dump(data, json_file)

def main():
    for dataset in DATASETS:
        for split in SPLITS:
            #break
            data_path = osp.join(MOT_DATA_ROOT, dataset, split)
            seqs = os.listdir(data_path)
            seqs = [s for s in seqs
                    if not s.endswith('GT') and not s.startswith('.') and not s.endswith('.json') and not s == 'reid']

            for seq in seqs:
                print(f"Processing sequence {seq} in dataset {dataset}")
                seq_path = osp.join(data_path, seq)
                seqinfo_path = osp.join(seq_path, 'seqinfo.ini')
                gt_path = osp.join(seq_path, 'gt/gt.txt')
                im_dir = osp.join(seq_path, 'img1')

                seqinfo = read_seqinfo(seqinfo_path)
                data = {'info': {'sequence': seq,
                                'dataset': dataset,
                                'split': split,
                                'creation_date': datetime.datetime.today().strftime('%Y-%m-%d-%H-%M'),
                                **seqinfo},
                        'images': [],
                        'annotations': [],
                        'categories': [{'id': 1, 'name': 'person', 'supercategory': 'person'}]}

                # Load Bounding Box annotations
                gt = np.loadtxt(gt_path, dtype=np.float32, delimiter=',')
                keep_classes = [1, 2, 7, 8, 12]
                mask = np.isin(gt[:, 7], keep_classes)
                gt = gt[mask]
                #break
                anns = [{'ped_id': row[1],
                        'frame_n': row[0],
                        'category_id': 1,
                        'id': f"{get_img_id(dataset, seq, f'{int(row[0]):06}.jpg')}{int(row_i):010}",
                        'image_id': get_img_id(dataset, seq, f'{int(row[0]):06}.jpg'),
                        'bbox': row[2:6].tolist(),
                        'area': row[4]*row[5],
                        'vis': row[8],
                        'iscrowd': 1 - row[6]}
                    for row_i, row in enumerate(gt.astype(float))]

                # Load Image information
                all_img_ids  =list(set([aa['image_id'] for aa in anns]))
                imgs = [{'file_name': osp.join(dataset, split, seq, 'img1', fname),
                        'height': seqinfo['height'],
                        'width': seqinfo['width'],
                        'id': get_img_id(dataset, seq, fname)
                        }
                        #for fname in os.listdir(im_dir)]
                        for fname in os.listdir(im_dir) if get_img_id(dataset, seq, fname) in all_img_ids]
                assert len(set([im['id'] for im in imgs]))  == len(imgs)
                data['images'].extend(imgs)


                #anns[0]['image_id']
                assert len(str(imgs[0]['id'])) == len(str(anns[0]['image_id']))

                data['annotations'].extend(anns)

                #ann_dir = osp.join(MOT_DATA_ROOT, 'annotations')
                ann_dir = osp.join(MOT_DATA_ROOT, ANN_DIR)
                ann_dir = data_path
                if not osp.exists(ann_dir):
                    os.makedirs(ann_dir)
                os.makedirs(ann_dir, exist_ok=True)
                # save_json(data, osp.join(ann_dir, f"{dataset}_{seq}.json"))
                save_json(data, osp.join(ann_dir, f"{seq}.json"))
                print(f"Saving result in {osp.join(ann_dir, f'{seq}.json')}")
                #cosa = read_json(osp.join(ann_dir, f"{dataset}_{seq}.json"))


if __name__ == '__main__':
    main()
