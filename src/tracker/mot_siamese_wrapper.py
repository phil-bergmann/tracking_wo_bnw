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

		self._train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10',
			'MOT17-11', 'MOT17-13']
		#self._train_folders = ["MOT17-09"]

		self._dataloader = MOT_Siamese(None, **dataloader)

		if image_set == "train":
			for seq in self._train_folders:
				d = MOT_Siamese(seq, **dataloader)
				for sample in d.data:
					self._dataloader.data.append(sample)
		else:
			raise NotImplementedError("Image set: {}".format(image_set))


	def __len__(self):
		return len(self._dataloader.data)

	def __getitem__(self, idx):
		return self._dataloader[idx]
