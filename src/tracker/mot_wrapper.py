from torch.utils.data import Dataset
import torch

from .mot_tracks import MOT_Tracks
from .mot_sequence import MOT_Sequence


class MOT_Wrapper(Dataset):
	"""Multiple Object Tracking Dataset.

	Wrapper class for combining (MOT_Sequence) or MOT_Tracks into one dataset
	"""

	def __init__(self, image_set, dataloader):
		self.prec_conv = False
		self.image_set = image_set

		self._train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
			'MOT17-11', 'MOT17-13']
		self._test_folders = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07',
			'MOT17-08', 'MOT17-12', 'MOT17-14']

		self._dataloader = MOT_Tracks(None, **dataloader)
		self.weights = []

		if image_set == "train":
			for seq in self._train_folders:
				d = MOT_Tracks(seq, **dataloader)
				for sample in d.data:
					self._dataloader.data.append(sample)
				self.weights += d.weights
		elif image_set == "test":
			for seq in self._test_folders:
				d = MOT_Tracks(seq, **dataloader)
				for sample in d.data:
					self._dataloader.data.append(sample)
				self.weights += d.weights
		else:
			raise NotImplementedError("Image set: {}".format(image_set))

	def precalculate_conv(self, frcnn):
		assert self.image_set == "train", "[!] Precalculating only implemented for train set not for: {}".format(self.image_set)
		self.prec_conv = True
		prec = {}
		for seq in self._train_folders:
			print("[*] Precalcuating conv of {}".format(seq))
			d = MOT_Sequence(seq)
			for sample in d:
				c = frcnn.get_net_conv(torch.from_numpy(sample['data']), torch.from_numpy(sample['im_info']))
				prec[sample['im_path']] = {'conv':c, 'im_info':sample['im_info']}

		self.prec_data = prec
		self._dataloader.generate_blobs = False

	def __len__(self):
		return len(self._dataloader.data)

	def __getitem__(self, idx):
		"""Return the ith Object"""
		if self.prec_conv:
			# add missing values and resize gt
			track = self._dataloader[idx]
			for f in track:
				prec = self.prec_data[f['im_path']]
				f['conv'] = prec['conv']
				f['im_info'] = prec['im_info']
				if 'gt' in f.keys():
					f['gt'] = f['gt'] * f['im_info'][2]
			return track
		else:
			return self._dataloader[idx]
