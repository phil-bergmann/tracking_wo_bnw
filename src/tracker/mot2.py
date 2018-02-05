from torch.utils.data import Dataset


class MOT(Dataset):
	"""Multiple Object Tracking Dataset."""

	def __init__(self, csv_file):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self._mot_dir = osp.join(cfg.DATA_DIR, 'MOT17Det')

		self._train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
			'MOT17-11', 'MOT17-13']
		self._test_folders = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07',
			'MOT17-08', 'MOT17-12', 'MOT17-14']

	def __len__(self):
		return len(self.landmarks_frame)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir,
								self.landmarks_frame.iloc[idx, 0])
		image = io.imread(img_name)
		landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
		landmarks = landmarks.astype('float').reshape(-1, 2)
		sample = {'image': image, 'landmarks': landmarks}

		if self.transform:
			sample = self.transform(sample)

		return sample

	def sequence(self, seq_name):
		if seq_name in self._train_folders:
			set_path = osp.join(self._mot_dir, 'train', seq_name)
		else:
			set_path = osp.join(self._mot_dir, 'test', seq_name)
			
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

		im_path = osp.join(set_path, imDir)
		gt_file = osp.join(set_path, 'gt', 'gt.txt')

		total = []
		train = []
		val = []

		for i in range(1,seqLength+1):
			im_path0 = osp.join(_imDir,"{:06d}.jpg".format(i))
			im_path1 = osp.join(_imDir,"{:06d}.jpg".format(i+1))

			d = { #'tracks':tracks,
				  'im_paths':[im_path0,im_path1],
			}
			total.append(d)
			if i <= seqLength*0.5:
				train.append(d)
			if i >= seqLength*0.75
				val.append(d)

		return total, train, val