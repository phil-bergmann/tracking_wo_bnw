import torch
from torch.utils.data import Dataset

from .mot_sequence import MOT17_Sequence, MOT19CVPR_Sequence, MOT17LOWFPS_Sequence


class MOT17_Wrapper(Dataset):
	"""A Wrapper for the MOT_Sequence class to return multiple sequences."""

	def __init__(self, split, dets, dataloader):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""
		train_sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
		test_sequences = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']

		if "train" == split:
			sequences = train_sequences
		elif "test" == split:
			sequences = test_sequences
		elif "all" == split:
			sequences = train_sequences + test_sequences
		elif f"MOT17-{split}" in train_sequences + test_sequences:
			sequences = [f"MOT17-{split}"]
		else:
			raise NotImplementedError("MOT split not available.")

		self._data = []
		for s in sequences:
			if dets == '17':
				self._data.append(MOT17_Sequence(seq_name=s, dets='DPM17', **dataloader))
				self._data.append(MOT17_Sequence(seq_name=s, dets='FRCNN17', **dataloader))
				self._data.append(MOT17_Sequence(seq_name=s, dets='SDP17', **dataloader))
			else:
				self._data.append(MOT17_Sequence(seq_name=s, dets=dets, **dataloader))

	def __len__(self):
		return len(self._data)

	def __getitem__(self, idx):
		return self._data[idx]


class MOT19CVPR_Wrapper(MOT17_Wrapper):
	"""A Wrapper for the MOT_Sequence class to return multiple sequences."""

	def __init__(self, split, dataloader):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""
		train_sequences = ['CVPR19-01', 'CVPR19-02', 'CVPR19-03', 'CVPR19-05']
		test_sequences = ['CVPR19-04', 'CVPR19-06', 'CVPR19-07', 'CVPR19-08']

		if "train" == split:
			sequences = train_sequences
		elif "test" == split:
			sequences = test_sequences
		elif "all" == split:
			sequences = train_sequences + test_sequences
		elif f"CVPR19-{split}" in train_sequences + test_sequences:
			sequences = [f"CVPR19-{split}"]
		else:
			raise NotImplementedError("MOT19CVPR split not available.")

		self._data = []
		for s in sequences:
			self._data.append(MOT19CVPR_Sequence(seq_name=s, **dataloader))

	def __len__(self):
		return len(self._data)

	def __getitem__(self, idx):
		return self._data[idx]


class MOT17LOWFPS_Wrapper(MOT17_Wrapper):
	"""A Wrapper for the MOT_Sequence class to return multiple sequences."""

	def __init__(self, split, dataloader):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""

		sequences = ['MOT17-02', 'MOT17-04', 'MOT17-09', 'MOT17-10', 'MOT17-11']

		self._data = []
		for s in sequences:
			self._data.append(MOT17LOWFPS_Sequence(split=split, seq_name=s, dets='FRCNN17', **dataloader))

	def __len__(self):
		return len(self._data)

	def __getitem__(self, idx):
		return self._data[idx]
