from torch.utils.data import Dataset, ConcatDataset

from .mot_reid import MOTreID


class MOTreIDWrapper(Dataset):
    """A Wrapper class for MOTSiamese.

    Wrapper class for combining different sequences into one dataset for the MOTreID
    Dataset.
    """

    def __init__(self, split, kwargs):
        train_sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
                           'MOT17-10', 'MOT17-11', 'MOT17-13']

        if split == "train":
            sequences = train_sequences
        elif f"MOT17-{split}" in train_sequences:
            sequences = [f"MOT17-{split}"]
        else:
            raise NotImplementedError("MOT split not available.")

        dataset = []
        for seq in sequences:
            # dataset.append(MOTreID(seq, split=split, **kwargs))
            dataset.append(MOTreID(seq, **kwargs))

        self.split = ConcatDataset(dataset)

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        return self.split[idx]
