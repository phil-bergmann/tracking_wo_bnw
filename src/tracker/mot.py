import os
import os.path as osp
import numpy as np
import csv
import configparser

from .config import cfg


class MOT(object):

	def __init__(self, image_set):

		self._image_set = image_set

		mot_dir = osp.join(cfg.DATA_DIR, 'MOT17')
		self._mot_train_dir = osp.join(mot_dir, 'train')
		self._mot_test_dir = osp.join(mot_dir, 'test')

		self._train_folders = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN',
			'MOT17-11-FRCNN', 'MOT17-13-FRCNN']
		self._test_folders = ['MOT17-01-FRCNN', 'MOT17-03-FRCNN', 'MOT17-06-FRCNN', 'MOT17-07-FRCNN',
			'MOT17-08-FRCNN', 'MOT17-12-FRCNN', 'MOT17-14-FRCNN']

		assert osp.exists(self._mot_train_dir), \
			'Path does not exist: {}'.format(self._mot_train_dir)
		assert osp.exists(self._mot_test_dir), \
			'Path does not exist: {}'.format(self._mot_test_dir)


		self._small_train = []
		self._small_val = []
		
		# for now find sequences of length 3 that contain good tracks
		for f in self._train_folders:
			sequence = f[:8]
			set_path = osp.join(self._mot_train_dir, f)
			config_file = osp.join(set_path, 'seqinfo.ini')

			assert osp.exists(config_file), \
				'Path does not exist: {}'.format(config_file)

			config = configparser.ConfigParser()
			config.read(config_file)
			seqLength = int(config['Sequence']['seqLength'])
			imWidth = int(config['Sequence']['imWidth'])
			imHeight = int(config['Sequence']['imHeight'])
			imExt = config['Sequence']['imExt']
			imDir = config['Sequence']['imDir']

			_imDir = osp.join(set_path, imDir)
			_gt_file = osp.join(set_path, 'gt', 'gt.txt')


			# gt data gt[image_num(starting at 1)][track_id]
			gt = {}
			for i in range(1, seqLength+1):
				gt[i] = {}

			with open(_gt_file, "r") as inf:
				reader = csv.reader(inf, delimiter=',')
				for row in reader:
					if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= 0.25:
						# Make pixel indexes 0-based, should already be 0-based (or not)
						x1 = int(row[2]) - 1
						y1 = int(row[3]) - 1
						# This -1 accounts for the width (width of 1 x1=x2)
						x2 = x1 + int(row[4]) - 1
						y2 = y1 + int(row[5]) - 1
						bb = [x1,y1,x2,y2]
						gt[int(row[0])][int(row[1])] = bb

			# mark all images i where there are good tracks until i+2
			good_track = []
			for i in range(1, seqLength-1):
				if gt[i].keys() == gt[i+1].keys():
					good_track.append(i)

			for i in good_track:
				im_path0 = osp.join(_imDir,"{:06d}.jpg".format(i))
				im_path1 = osp.join(_imDir,"{:06d}.jpg".format(i+1))

				# all should be of same length
				num_objs = len(gt[i])
				# x1,y1,x2,y2
				# (id,frame,bb)
				tracks = np.zeros((num_objs,3,4), dtype=np.float32)
				for j,id in enumerate(gt[i]):
					tracks[j,0,:] = gt[i][id]
					tracks[j,1,:] = gt[i+1][id]

				d = { 'tracks':tracks,
					  'im_paths':[im_path0,im_path1],
				}
				if i <= seqLength*0.5:
					self._small_train.append(d)
				if i > seqLength*0.75:
					self._small_val.append(d)

		if self._image_set == "small_train":
			self._db = self._small_train
		elif self._image_set == "small_val":
			self._db = self._small_val


	def get_data(self):
		return self._db




