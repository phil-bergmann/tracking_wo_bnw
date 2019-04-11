import os
import os.path as osp
import sys

from PIL import Image

from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets

dataset = "mot_train_"
detections = "FRCNN"

module_dir = get_output_dir('MOT17')
results_dir = module_dir
module_dir = osp.join(module_dir, 'eval/video_fp')

tracker = ["Tracktor", "FWT", "jCC", "MOTDT17"]


for db in Datasets(dataset):
    seq_path = osp.join(module_dir, f"{tracker[0]}/{db}-{detections}")
    if not osp.exists(seq_path):
        continue
    for frame, v in enumerate(db, 1):
        file_name = osp.basename(v['im_path'])
        output_dir = osp.join(module_dir, 'combined', f"{db}-{detections}")
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        im_output = osp.join(output_dir, file_name)

        tracker_frames = []
        for t in tracker:
            im_path = osp.join(module_dir, f"{t}/{db}-{detections}/{file_name}")
            tracker_frames.append(Image.open(im_path))

        width = tracker_frames[0].size[0]
        height = tracker_frames[0].size[1]

        combined_frame = Image.new('RGB', (width * 2, height * 2))
        combined_frame.paste(tracker_frames[0], (0, 0))
        combined_frame.paste(tracker_frames[1], (width, 0))
        combined_frame.paste(tracker_frames[2], (0, height))
        combined_frame.paste(tracker_frames[3], (width, height))

        combined_frame.save(im_output)
