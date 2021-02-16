
from torch.utils.data import ConcatDataset

from .marcuhmot import MarCUHMOT
from .mot15_wrapper import MOT15Wrapper
from .mot_reid_wrapper import MOTreIDWrapper
from .mot_wrapper import (MOT17LOWFPSWrapper, MOT17PrivateWrapper,
                          MOT17Wrapper, MOT19Wrapper, MOT20Wrapper)

_sets = {}


# Fill all available datasets, change here to modify / add new datasets.
for split in ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08', '09',
              '10', '11', '12', '13', '14']:
    for dets in ['DPM16', 'DPM', 'FRCNN', 'SDP', 'ALL']:
        name = f'mot17_{split}_{dets}'
        _sets[name] = (lambda *args, split=split,
                       dets=dets: MOT17Wrapper(split, dets, *args))

for split in ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08']:
    # only FRCNN detections
    name = f'mot19_{split}'
    _sets[name] = (lambda *args, split=split: MOT19Wrapper(split, *args))

for split in ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08']:
    # only FRCNN detections
    name = f'mot20_{split}'
    _sets[name] = (lambda *args, split=split: MOT20Wrapper(split, *args))

for split in ['1', '2', '3', '5', '6', '10', '15', '30']:
    # only FRCNN detections
    name = f'mot17_{split}_fps'
    _sets[name] = (lambda *args, split=split: MOT17LOWFPSWrapper(split, *args))

# for split in ['train', 'small_val', 'small_train']:
for split in ['train', '02', '04', '05', '09', '10', '11','13',]:
    name = f'mot_reid_{split}'
    _sets[name] = (lambda *args, split=split: MOTreIDWrapper(split, *args))

for split in ['PETS09-S2L1', 'TUD-Stadtmitte', 'TUD-Campus', 'train', 'test', 'last3train']:
    name = f'mot15_{split}'
    _sets[name] = (lambda *args, split=split: MOT15Wrapper(split, *args))

for split in ['small_train', 'small_val', 'train']:
    name = f'marcuhmot_{split}'
    _sets[name] = (lambda *args, split=split: MarCUHMOT(split, *args))


class Datasets(object):
    """A central class to manage the individual dataset loaders.

    This class contains the datasets. Once initialized the individual parts (e.g. sequences)
    can be accessed.
    """

    def __init__(self, datasets, *args):
        """Initialize the corresponding dataloader.

        Keyword arguments:
        dataset --  the name of the dataset
        args -- arguments used to call the dataloader
        """
        if isinstance(datasets, str):
            datasets = [datasets]

        if len(args) == 0:
            args = [{}]

        self.datasets = None
        for dataset in datasets:
            assert dataset in _sets, f"[!] Dataset not found: {dataset}"

            if self.datasets is None:
                self.datasets = _sets[dataset](*args)
            else:
                self.datasets = ConcatDataset([self.datasets, _sets[dataset](*args)])

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]
