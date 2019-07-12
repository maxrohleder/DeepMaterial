from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np
import torch

# make sure this value fits the one in mnistToNumpy.py (or count files in dirs)
MAX_FILES_PER_FOLDER = 10000


def recursiveCount(start):
    sum = 0
    for _, directories, filenames in os.walk(start):
        for d in directories:
            sum += recursiveCount(d)
        sum += len(filenames)
    return sum


def getSubdir(index, n):
    if n <= MAX_FILES_PER_FOLDER:
        return ''
    return str(int(index/10000)*10+10) + 'k'


class MnistDataset(Dataset):

    def __init__(self, root_dir, train=True, transform=None):
        if train:
            self.root_dir = os.path.join(root_dir, 'train')
            self.prefix = 'train'
        else:
            self.root_dir = os.path.join(root_dir, 'test')
            self.prefix = 'test'
        self.number_of_files = recursiveCount(self.root_dir)
        self.transform = transform

    def __getitem__(self, index):
        if index < 0 or index >= self.number_of_files:
            raise IndexError
        path_string = self.root_dir + '/**/' + self.prefix + '_' + str(index) + '_*.npy'
        file = glob.glob(path_string, recursive=True).pop()
        label = file[-5:-4]
        sample = {'image': np.load(file), 'label': int(label)}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.number_of_files


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image.reshape(1, 28, 28)
        return {'image': torch.from_numpy(image),
                'label': torch.scalar_tensor(label)}


if __name__ == "__main__":
    training_set = MnistDataset("data/numpy/", train=True)
    print("training set length: ", len(training_set))
    test_set = MnistDataset("data/numpy", train=False)
    print("test set length: ", len(test_set))