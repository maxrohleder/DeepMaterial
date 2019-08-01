from __future__ import print_function, division
import os
import shutil
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import trange, tqdm
from matplotlib import pyplot as plt
import pyconrad
import time


class DectDataset(Dataset):
    """Zeego projection space DECT dataset """

    def __init__(self, root_dir, device='cpu', precompute=False, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.precompute=precompute
        self.torchDump =os.path.join(root_dir, "torch")
        self.device = device
        self.length = sum(os.path.isdir(os.path.join(root_dir, i)) and "_" in i for i in os.listdir(root_dir))
        self.numProjections = 200
        self.transform = transform
        # detect number of samples
        self.samples = []
        for i in range(self.length):
            naming_scheme = "{}/*_{}/**/*.raw".format(root_dir, i)
            self.samples.append([n for n in glob.glob(naming_scheme, recursive=True)])

        if precompute:
            if not os.path.isdir(self.torchDump):
                print("building a faster accessible version of the dataset")
                os.makedirs(self.torchDump)
                self.constructTorchDump(self.torchDump)
            elif os.path.isdir(self.torchDump) and len(os.listdir(self.torchDump)) != self.length * 2 * self.numProjections:
                print("torchdump seems outdated.. deleting and rebuilding...")
                shutil.rmtree(self.torchDump)
                os.makedirs(self.torchDump)
                self.constructTorchDump(self.torchDump)
            else:
                print("using existing dump at: ", self.torchDump)

    def __len__(self):
        return self.length * self.numProjections

    def __getitem__(self, idx):
        if self.precompute:
            return torch.load(self.torchDump+"/poly_{}.pt".format(idx)), torch.load(self.torchDump+"/mat_{}.pt".format(idx))
        else:
            files = self.samples[int((idx/(self.numProjections*self.length))*4)]
            poly, mat = self.readToNumpy(files)
            mat = mat[:, idx%200, :, :].reshape(3, 480, 620)
            poly = poly[:, idx%200, :, :].reshape(2, 480, 620)

            mat = torch.tensor(mat, device=self.device)
            poly = torch.tensor(poly, device=self.device)
            return poly, mat

    def readToNumpy(self, files):
        '''
        reads in big endian 32 bit float raw files and outputs numpy data
        :param files:
        :return:
        '''
        assert len(files) == 5
        dt = np.dtype('>f4')
        poly = np.empty((2, 200, 480, 620))
        mat = np.empty((3, 200, 480, 620))
        mat_channel = 0
        for file in files:
            if 'POLY80' in file:
                f = np.fromfile(file=file, dtype=dt).newbyteorder().byteswap()
                poly[0] = f.reshape(200, 480, 620)
            elif 'POLY120' in file:
                f = np.fromfile(file=file, dtype=dt).newbyteorder().byteswap()
                poly[1] = f.reshape(200, 480, 620)
            elif 'MAT' in file:
                f = np.fromfile(file=file, dtype=dt).newbyteorder().byteswap()
                mat[mat_channel] = f.reshape(200, 480, 620)
                mat_channel += 1
            else:
                print("ERROR IN FILE STRUCTURE")
        return poly, mat

    def constructTorchDump(self, path):
        print("precomputing dataset to: ", self.torchDump)
        for i, sample in enumerate(tqdm(self.samples)):
            poly, mat = self.readToNumpy(sample)
            for p in range(self.numProjections):
                matp = mat[:, p, :, :].reshape(3, 480, 620)
                polyp = poly[:, p, :, :].reshape(2, 480, 620)
                matp = torch.tensor(matp, device=self.device)
                polyp = torch.tensor(polyp, device=self.device)
                torch.save(matp, self.torchDump + "/mat_{}.pt".format((i * self.numProjections) + p))
                torch.save(polyp, self.torchDump + "/poly_{}.pt".format((i * self.numProjections) + p))


if __name__ == "__main__":

    #use_cuda = torch.cuda.is_available()
    use_cuda = False
    device = torch.device("cuda:0" if use_cuda else "cpu")

    dp = DectDataset("/home/mr/Documents/bachelor/data/simulation", precompute=False, device=device)
    dnp = DectDataset("/home/mr/Documents/bachelor/data/simulation", precompute=True, device=device)

    set_pre = DataLoader(dp, batch_size=50, num_workers=10, shuffle=False)
    set_np = DataLoader(dnp, batch_size=50, num_workers=10, shuffle=False)

    start = time.time()
    for x, y in tqdm(set_pre):
        x, y = x.to(device), y.to(device)
        pass
    d1 = time.time() - start

    start = time.time()
    for x, y in tqdm(set_np):
        x, y = x.to(device), y.to(device)
        pass
    d2 = time.time() - start

    print("pre: {}\nnpre: {}\nratio: {}".format(d1, d2, d1/d2))
