import torch
from torch.utils.data import Dataset

from .jrdb_sequence import JRDB_Sequence


class JRDB_Wrapper(Dataset):
	"""A Wrapper for the JRDB_Sequence class to return multiple sequences."""

	def __init__(self, split, dataloader):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""
		train_sequences = ['bytes-cafe-2019-02-07_0', 'gates-159-group-meeting-2019-04-03_0', 'huang-basement-2019-01-25_0', 'packard-poster-session-2019-03-20_0', 'tressider-2019-03-16_0', 'clark-center-2019-02-28_0', 'gates-ai-lab-2019-02-08_0', 'huang-lane-2019-02-12_0', 'packard-poster-session-2019-03-20_1', 'tressider-2019-03-16_1', 'clark-center-2019-02-28_1', 'gates-basement-elevators-2019-01-17_1', 'jordan-hall-2019-04-22_0', 'packard-poster-session-2019-03-20_2', 'tressider-2019-04-26_2', 'clark-center-intersection-2019-02-28_0', 'gates-to-clark-2019-02-28_1', 'memorial-court-2019-03-16_0', 'stlc-111-2019-04-19_0', 'cubberly-auditorium-2019-04-22_0', 'hewlett-packard-intersection-2019-01-24_0', 'meyer-green-2019-03-16_0', 'svl-meeting-gates-2-2019-04-08_0', 'forbes-cafe-2019-01-22_0', 'huang-2-2019-01-25_0','nvidia-aud-2019-04-18_0', 'svl-meeting-gates-2-2019-04-08_1']
		test_sequences = ['cubberly-auditorium-2019-04-22_1', 'gates-basement-elevators-2019-01-17_0', 'huang-2-2019-01-25_1', 'nvidia-aud-2019-01-25_0', 'serra-street-2019-01-30_0', 'tressider-2019-04-26_1', 'discovery-walk-2019-02-28_0', 'gates-foyer-2019-01-17_0', 'huang-intersection-2019-01-22_0', 'nvidia-aud-2019-04-18_1', 'stlc-111-2019-04-19_1', 'tressider-2019-04-26_3', 'discovery-walk-2019-02-28_1', 'gates-to-clark-2019-02-28_0', 'indoor-coupa-cafe-2019-02-06_0', 'nvidia-aud-2019-04-18_2', 'stlc-111-2019-04-19_2', 'food-trucks-2019-02-12_0', 'hewlett-class-2019-01-23_0', 'lomita-serra-intersection-2019-01-30_0', 'outdoor-coupa-cafe-2019-02-06_0', 'tressider-2019-03-16_2', 'gates-ai-lab-2019-04-17_0', 'hewlett-class-2019-01-23_1', 'meyer-green-2019-03-16_1', 'quarry-road-2019-02-28_0', 'tressider-2019-04-26_0']

		if "train" == split:
			sequences = train_sequences
		elif "test" == split:
			sequences = test_sequences
		elif "all" == split:
			sequences = train_sequences + test_sequences
		else:
			raise NotImplementedError("MOT split not available.")

		self._data = []
		for s in sequences:
			self._data.append(JRDB_Sequence(seq_name=s, dets='det', **dataloader))

	def __len__(self):
		return len(self._data)

	def __getitem__(self, idx):
		return self._data[idx]


