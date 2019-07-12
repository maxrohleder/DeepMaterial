from __future__ import print_function, division
import os
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader


class DectDataset(Dataset):
    """Zeego projection space DECT dataset """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.number_of_files = len([name for name in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, name))])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

def recursive_count(dir):
    for filename in os.listdir(dir):
        if os.path.isdir(filename):
            recursive_count(dir)
        else:
