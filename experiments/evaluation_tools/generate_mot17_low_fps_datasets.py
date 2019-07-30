import os
import shutil
from configparser import ConfigParser


if __name__ == "__main__":
    data_dir = "data/MOT17_LOW_FPS"
    labels_dir = "/storage/slurm/meinhard/data/MOT17Labels/train/"
    imgs_dir = "/storage/slurm/meinhard/data/MOT17Det/train/"

    frame_skips = [1, 2, 3, 5, 6, 10, 15, 30]

    train_sequences = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN"]

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    for f_s in frame_skips:
        dataset_dir = os.path.join(data_dir, f"MOT17_{30 // f_s}_FPS")
        os.makedirs(dataset_dir)

        # TEST
        test_dataset_dir = os.path.join(dataset_dir, 'test')
        os.makedirs(test_dataset_dir)

        # TRAIN
        train_dataset_dir = os.path.join(dataset_dir, 'train')
        os.makedirs(train_dataset_dir)
        for s in train_sequences:
            dataset_seq_dir = os.path.join(train_dataset_dir, s)
            os.makedirs(dataset_seq_dir)

            #DET
            os.makedirs(os.path.join(dataset_seq_dir, 'det'))

            det_file_path = os.path.join(labels_dir, s, 'det/det.txt')
            new_det_file_path = os.path.join(dataset_seq_dir, 'det/det.txt')

            f = open(det_file_path, 'r')
            linelist = f.readlines()
            f.close

            f2 = open(new_det_file_path, 'w')
            for line in linelist:
                line_split = line.split(',')
                frame_id = int(line_split[0])
                if frame_id % f_s == 0:
                    new_frame_id = frame_id // f_s
                    line_split[0] = str(new_frame_id)
                    new_line = ','.join(line_split)
                    f2.write(new_line)
            f2.close()

            #GT
            os.makedirs(os.path.join(dataset_seq_dir, 'gt'))

            gt_file_path = os.path.join(labels_dir, s, 'gt/gt.txt')
            new_gt_file_path = os.path.join(dataset_seq_dir, 'gt/gt.txt')

            f = open(gt_file_path, 'r')
            linelist = f.readlines()
            f.close

            f2 = open(new_gt_file_path, 'w')
            for line in linelist:
                line_split = line.split(',')
                frame_id = int(line_split[0])
                if frame_id % f_s == 0:
                    new_frame_id = frame_id // f_s
                    line_split[0] = str(new_frame_id)
                    new_line = ','.join(line_split)
                    f2.write(new_line)
            f2.close()

            #IMG
            dataset_seq_dir_img = os.path.join(train_dataset_dir, s.replace('-FRCNN', ''))
            os.makedirs(dataset_seq_dir_img)
            os.makedirs(os.path.join(dataset_seq_dir_img, 'img1'))

            img_dir = os.path.join(imgs_dir, s.replace('-FRCNN', ''), 'img1')

            num_frames = 0
            for img_file_name in os.listdir(img_dir):
                if '_' not in img_file_name:
                    frame_id = int(img_file_name.split('.')[0])
                    if frame_id % f_s == 0:
                        new_frame_id = frame_id // f_s

                        os.symlink(os.path.join(img_dir, img_file_name),
                                   os.path.join(dataset_seq_dir_img, 'img1', f"{new_frame_id:06d}.jpg"))

                        # shutil.copyfile(os.path.join(img_dir, img_file_name),
                        #                 os.path.join(dataset_seq_dir_img, 'img1', f"{new_frame_id:06d}.jpg"))

                        num_frames += 1

            #CONFIG
            shutil.copyfile(os.path.join(labels_dir, s, 'seqinfo.ini'),
                            os.path.join(dataset_seq_dir_img, 'seqinfo.ini'))

            parser = ConfigParser()
            parser.read(os.path.join(dataset_seq_dir_img, 'seqinfo.ini'))

            parser['Sequence']['seqLength'] = str(num_frames)

            with open(os.path.join(dataset_seq_dir_img, 'seqinfo.ini'), 'w') as configfile:
                parser.write(configfile)
