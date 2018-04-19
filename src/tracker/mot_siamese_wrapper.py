from torch.utils.data import Dataset
import torch

from .mot_siamese import MOT_Siamese


class MOT_Siamese_Wrapper(Dataset):
	"""Multiple Object Tracking Dataset.

	Wrapper class for combining different sequences into one dataset for the MOT_Siamese
	Dataset.
	"""

	def __init__(self, image_set, dataloader):
		self.prec_conv = False
		self.image_set = image_set

		if image_set == "train":
			self._train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
				'MOT17-11', 'MOT17-13']
		elif image_set == "mot-09":
			self._train_folders = ["MOT17-09"]
		else:
			raise NotImplementedError("Image set: {}".format(image_set))

		self._dataloader = MOT_Siamese(None, **dataloader)

		for seq in self._train_folders:
			d = MOT_Siamese(seq, **dataloader)
			for sample in d.data:
				self._dataloader.data.append(sample)

	def __len__(self):
		return len(self._dataloader.data)

	def __getitem__(self, idx):
		return self._dataloader[idx]
