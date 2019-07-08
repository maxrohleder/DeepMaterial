from torch.utils.data import Dataset, DataLoader


class MnistDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
