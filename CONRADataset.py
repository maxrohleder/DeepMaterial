from __future__ import print_function, division
import os
import shutil
import glob
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import time

from createTorchDump import constructTorchDump
from models.unet import UNet


class CONRADataset(Dataset):
    """Zeego projection space DECT dataset """

    def __init__(self, root_dir, train, device='cpu', precompute=False, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.root_dir = root_dir
        self.precompute = precompute
        self.torchDump = os.path.join(root_dir, "torch")
        self.device = torch.device(device)
        self.length = sum(os.path.isdir(os.path.join(root_dir, d)) and "_" in d for d in os.listdir(root_dir))
        self.transform = transform
        # detect number of samples
        self.samples = []
        for i in range(self.length):
            naming_scheme = "{}/*_{}/**/*.raw".format(root_dir, i)
            self.samples.append([n for n in glob.glob(naming_scheme, recursive=True)])
        self.X, self.Y, self.P = self.detectSizesFromFilename(self.samples[0][0])
        # Dual Energy implicates 2 energy projections (rest must be material)
        self.numberOfMaterials = len(self.samples[0]) - 2

        if precompute:
            if not os.path.isdir(self.torchDump):
                print("building a faster accessible version of the dataset")
                os.makedirs(self.torchDump)
                constructTorchDump(root_dir, self.torchDump, True, device)
            elif os.path.isdir(self.torchDump) and len(
                    glob.glob(self.torchDump + "/**/*.pt", recursive=True)) != self.length * 2 * self.P:
                print("torchdump seems outdated.. deleting and rebuilding...")
                shutil.rmtree(self.torchDump)
                os.makedirs(self.torchDump)
                constructTorchDump(root_dir, self.torchDump, True, device)
            else:
                print("using existing dump at: ", self.torchDump)
            # select train or test folder in torchdump
            if train:
                self.torchDump += "/train"
            else:
                self.torchDump += '/test'

    def __len__(self):
        return self.length * self.P

    def __getitem__(self, idx):
        if self.precompute:
            X, Y = torch.load(self.torchDump + "/poly_{}.pt".format(idx)), \
                   torch.load(self.torchDump + "/mat_{}.pt".format(idx))
            if self.transform is not None:
                return self.transform(X), self.transform(Y)
            else:
                return X, Y
        else:
            files = self.samples[int((idx / (self.P * self.length)) * 4)]
            poly, mat = self.readToNumpy(files)
            mat = mat[:, idx % 200, :, :].reshape(3, self.Y, self.X)
            poly = poly[:, idx % 200, :, :].reshape(2, self.Y, self.X)

            mat = torch.tensor(mat, device=self.device)
            poly = torch.tensor(poly, device=self.device)
            return poly, mat

    def readToNumpy(self, files):
        '''
        reads in big endian 32 bit float raw files and outputs numpy data
        :param files: string array containing the paths to poly and mat files
        :return: numpy arrays in format [C, P, Y, X] C = number of materials/energy projections
        '''
        dt = np.dtype('>f4')
        poly = np.empty((2, 200, self.Y, self.X))
        mat = np.empty((self.numberOfMaterials, 200, self.Y, self.X))
        mat_channel = 0
        for file in files:
            if 'POLY80' in file:
                f = np.fromfile(file=file, dtype=dt).newbyteorder().byteswap()
                poly[0] = f.reshape(200, self.Y, self.X)
            elif 'POLY120' in file:
                f = np.fromfile(file=file, dtype=dt).newbyteorder().byteswap()
                poly[1] = f.reshape(200, self.Y, self.X)
            elif 'MAT' in file:
                f = np.fromfile(file=file, dtype=dt).newbyteorder().byteswap()
                mat[mat_channel] = f.reshape(200, self.Y, self.X)
                mat_channel += 1
            else:
                print("ERROR IN FILE STRUCTURE")
        return poly, mat

    def detectSizesFromFilename(self, filename):
        pa, filename = os.path.split(filename)
        fle, ext = os.path.splitext(filename)
        parts = fle.split("x")
        X = int(parts[0].split('_')[1])
        Y = int(parts[1])
        P = int(parts[2].split('.')[0])
        return X, Y, P

if __name__ == "__main__":

    # use_cuda = torch.cuda.is_available()
    use_cuda = False
    device = "cuda:0" if use_cuda else "cpu"

    dp = CONRADataset("/home/mr/Documents/bachelor/data/simulation", True, precompute=True, device=device)
    dnp = CONRADataset("/home/mr/Documents/bachelor/data/simulation", False, precompute=True, device=device)

    a, b = dp[0]

    trainset = DataLoader(dp, batch_size=2, num_workers=5, shuffle=True)
    testset = DataLoader(dnp, batch_size=2, num_workers=5, shuffle=False)

    m = UNet(2, 2).to(device)
    loss_fn = nn.BCELoss(reduction='sum')

    m.eval()
    with torch.no_grad():
        test_loss = 0
        for x, y in tqdm(testset):
            x, y = x.to(device), y.to(device)
            pred = m(x)
            test_loss += loss_fn(pred, y).item()

        train_loss = 0
        for x, y in tqdm(testset):
            x, y = x.to(device), y.to(device)
            pred = m(x)
            train_loss += loss_fn(pred, y).item()

        print("initial loss is {}/{} (train/test)".format(train_loss, test_loss))

    # start = time.time()
    # for x, y in tqdm(set_pre):
    #     x, y = x.to(device), y.to(device)
    #     pass
    # d1 = time.time() - start
    #
    # start = time.time()
    # for x, y in tqdm(set_np):
    #     x, y = x.to(device), y.to(device)
    #     pass
    # d2 = time.time() - start
    #
    # print("pre: {}\nnpre: {}\nratio: {}".format(d1, d2, d1 / d2))
