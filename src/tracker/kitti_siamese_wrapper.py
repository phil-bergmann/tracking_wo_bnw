from torch.utils.data import Dataset
import torch

from .kitti_siamese import KITTI_Siamese


class KITTI_Siamese_Wrapper(Dataset):
	"""Multiple Object Tracking Dataset.

	Wrapper class for combining different sequences into one dataset for the MOT_Siamese
	Dataset.
	"""

	def __init__(self, image_set, dataloader):
		self.image_set = image_set

		if "train_Car" in image_set:
			self._train_folders = ["train_%04d_Car"%(seq) for seq in range(21)]
		elif "train_Pedestrian" in image_set:
			self._train_folders = ["train_%04d_Pedestrian"%(seq) for seq in range(21)]
		else:
			raise NotImplementedError("Image set: {}".format(image_set))

		self._dataloader = KITTI_Siamese(None, **dataloader)

		for seq in self._train_folders:
			d = KITTI_Siamese(seq, **dataloader)
			for sample in d.data:
				self._dataloader.data.append(sample)

	def __len__(self):
		return len(self._dataloader.data)

	def __getitem__(self, idx):
		return self._dataloader[idx]
