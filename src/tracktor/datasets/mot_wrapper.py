from torch.utils.data import Dataset
import torch

from .mot_sequence import MOT_Sequence


class MOT_Wrapper(Dataset):
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
		elif f"MOT17-{split}" in train_sequences:
			sequences = [train_sequences[train_sequences.index(f"MOT17-{split}")]]
		elif f"MOT17-{split}" in test_sequences:
			sequences = [test_sequences[test_sequences.index(f"MOT17-{split}")]]
		else:
			raise NotImplementedError("MOT split not available.")

		self._data = []
		for s in sequences:
			self._data.append(MOT_Sequence(seq_name=s, dets=dets, **dataloader))

	def __len__(self):
		return len(self._data)

	def __getitem__(self, idx):
		return self._data[idx]
