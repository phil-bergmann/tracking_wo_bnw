
from .mot_wrapper import MOT_Wrapper
from .mot_siamese_wrapper import MOT_Siamese_Wrapper
from .mot15_wrapper import MOT15_Wrapper
from .marcuhmot import MarCUHMOT


_sets = {}


# Fill all available datasets, change here to modify / add new datasets.
for split in ['train', 'test', '01', '02', '03', '04', '05', '06', '07', '08', '09',
			  '10', '11', '12', '13', '14']:
	for dets in ['DPM16', 'DPM_RAW16', 'DPM17', 'FRCNN17', 'SDP17', '']:
		name = 'mot_{}_{}'.format(split, dets)
		_sets[name] = (lambda *args, split=split, dets=dets: MOT_Wrapper(split, dets, *args))

for split in ['train', 'smallVal', 'smallTrain']:
	name = 'motSiamese_{}'.format(split)
	_sets[name] = (lambda *args, split=split: MOT_Siamese_Wrapper(split, *args))

for split in ['PETS09-S2L1', 'TUD-Stadtmitte', 'TUD-Campus', 'train', 'test', 'last3train']:
	name = 'mot15_{}'.format(split)
	_sets[name] = (lambda *args, split=split: MOT15_Wrapper(split, *args))

for split in ['smallTrain', 'smallVal', 'train']:
	name = 'marcuhmot_{}'.format(split)
	_sets[name] = (lambda *args, split=split: MarCUHMOT(split, *args))


class Datasets(object):
	"""A central class to manage the individual dataset loaders.

	This class contains the datasets. Once initialized the individual parts (e.g. sequences)
	can be accessed.
	"""

	def __init__(self, dataset, *args):
		"""Initialize the corresponding dataloader.

		Keyword arguments:
		dataset --  the name of the dataset
		args -- arguments used to call the dataloader
		"""
		assert dataset in _sets, "[!] Dataset not found: {}".format(dataset)

		if len(args) == 0:
			args = [{}]

		self._data = _sets[dataset](*args)

	def __len__(self):
		return len(self._data)

	def __getitem__(self, idx):
		return self._data[idx]

		
