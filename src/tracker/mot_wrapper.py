from torch.utils.data import Dataset

from .mot_tracks import MOT_Tracks


class MOT_Wrapper(Dataset):
	"""Multiple Object Tracking Dataset.

	Wrapper class for combining (MOT_Sequence) or MOT_Tracks into one dataset
	"""

	def __init__(self, image_set, dataloader=MOT_Tracks):

		self._train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
			'MOT17-11', 'MOT17-13']
		self._test_folders = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07',
			'MOT17-08', 'MOT17-12', 'MOT17-14']

		assert image_set in ["train", "test"], "[!] Invalid image set: {}".format(image_set)

		self._dataloader = dataloader()

		if image_set == "train":
			for seq in self._train_folders:
				d = dataloader(seq)
				for sample in d.data:
					self._dataloader.data.append(sample)
		if image_set == "test":
			for seq in self._test_folders:
				d = dataloader(seq)
				for sample in d.data:
					self._dataloader.data.append(sample)

	def __len__(self):
		return len(self._dataloader.data)

	def __getitem__(self, idx):
		"""Return the ith Object"""
		return self._dataloader[idx]