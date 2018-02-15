import os
import os.path as osp
import numpy as np
import csv
import configparser

from .config import cfg


class MOT(object):

	def __init__(self, image_set):

		self._image_set = image_set

		self._db = []

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


		if "small" in self._image_set:
			self.small()
		# here only one sequence is requested in full, not gt needed
		elif "MOT" in self._image_set:
			self.sequence()



	def sequence(self):
		self._db = []

		if self._image_set in self._train_folders:
			set_path = osp.join(self._mot_train_dir, self._image_set)
		else:
			set_path = osp.join(self._mot_test_dir, self._image_set)
			
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

		for i in range(1,seqLength+1):
			im_path0 = osp.join(_imDir,"{:06d}.jpg".format(i))
			im_path1 = osp.join(_imDir,"{:06d}.jpg".format(i+1))

			d = { #'tracks':tracks,
				  'im_paths':[im_path0,im_path1],
			}
			self._db.append(d)
		
		
	def small(self):
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

			# mark all images i where there are good tracks until i+1
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
				tracks = np.zeros((num_objs,2,4), dtype=np.float32)
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

	@property
	def size(self):
		return len(self._db)

	def image_paths_at(self, i):
		return self._db[i]['im_paths']

	@property
	def data(self):
		return self._db


	def _write_results_file(self, all_tracks, output_dir):
		"""Write the tracks in the format for MOT16/MOT17 sumbission

		all_tracks: list with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...}

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

		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		file = osp.join(output_dir, 'MOT16-'+self._image_set[6:8]+'.txt')

		print("[*] Writing to: {}".format(file))

		with open(file, "w") as of:
			writer = csv.writer(of, delimiter=',')
			for i, track in enumerate(all_tracks):
				for frame, bb in track.items():
					x1 = bb[0]
					y1 = bb[1]
					x2 = bb[2]
					y2 = bb[3]
					writer.writerow([frame, i, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])



