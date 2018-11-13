from torch.utils.data import Dataset

from .mot15_sequence import MOT15_Sequence


class MOT15_Wrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split, dataloader):
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        dataloader -- args for the MOT_Sequence dataloader
        """

        train_sequences = ['Venice-2', 'KITTI-17', 'KITTI-13', 'ADL-Rundle-8', 'ADL-Rundle-6', 'ETH-Pedcross2',
                           'ETH-Sunnyday', 'ETH-Bahnhof', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte']
        test_sequences = ['Venice-1', 'KITTI-19', 'KITTI-16', 'ADL-Rundle-3', 'ADL-Rundle-1', 'AVG-TownCentre',
                          'ETH-Crossing', 'ETH-Linthescher', 'ETH-Jelmoli', 'PETS09-S2L2', 'TUD-Crossing']

        if "train" == split:
            sequences = train_sequences
        elif "test" == split:
            sequences = test_sequences
        elif "last3train" == split:
            sequences = train_sequences[-3:]
        elif split in train_sequences or split in test_sequences:
            sequences = [split]
        else:
            raise NotImplementedError("Image set: {}".format(split))

        self._data = []

        for s in sequences:
            self._data.append(MOT15_Sequence(seq_name=s, **dataloader))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
