from torch.utils.data import Dataset, ConcatDataset

from .mot_reid import MOTreID


class MOTreIDWrapper(Dataset):
    """A Wrapper class for MOTSiamese.

    Wrapper class for combining different sequences into one dataset for the MOTreID
    Dataset.
    """

    def __init__(self, split, dataloader):
        mot_dir = 'MOT17'
        train_folders = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN',
                         'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN']

        dataset = []
        for seq in train_folders:
            dataset.append(MOTreID(seq, mot_dir, split=split, **dataloader))

        self._dataset = ConcatDataset(dataset)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]
