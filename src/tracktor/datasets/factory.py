
from .mot_wrapper import MOT17_Wrapper, MOT19CVPR_Wrapper, MOT17LOWFPS_Wrapper
from .mot_siamese_wrapper import MOT_Siamese_Wrapper
from .mot15_wrapper import MOT15_Wrapper
from .marcuhmot import MarCUHMOT


_sets = {}


# Fill all available datasets, change here to modify / add new datasets.
for split in ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08', '09',
              '10', '11', '12', '13', '14']:
    for dets in ['DPM16', 'DPM_RAW16', 'DPM17', 'FRCNN17', 'SDP17', '17', '']:
        name = f'mot17_{split}_{dets}'
        _sets[name] = (lambda *args, split=split,
                       dets=dets: MOT17_Wrapper(split, dets, *args))

for split in ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08']:
    # only FRCNN detections
    name = f'mot19_cvpr_{split}'
    _sets[name] = (lambda *args, split=split: MOT19CVPR_Wrapper(split, *args))

for split in ['1', '2', '3', '5', '6', '10', '15', '30']:
    # only FRCNN detections
    name = f'mot17_{split}_fps'
    _sets[name] = (lambda *args, split=split: MOT17LOWFPS_Wrapper(split, *args))

for split in ['train', 'smallVal', 'smallTrain']:
    name = f'motSiamese_{split}'
    _sets[name] = (lambda *args, split=split: MOT_Siamese_Wrapper(split, *args))

for split in ['PETS09-S2L1', 'TUD-Stadtmitte', 'TUD-Campus', 'train', 'test', 'last3train']:
    name = f'mot15_{split}'
    _sets[name] = (lambda *args, split=split: MOT15_Wrapper(split, *args))

for split in ['smallTrain', 'smallVal', 'train']:
    name = f'marcuhmot_{split}'
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
