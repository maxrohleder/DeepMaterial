import torch
import os
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from CONRADataset import CONRADataset
from models.unet import UNet

from skimage.measure import compare_ssim as ssim


"""
------------------------------ params ------------------------------
"""

DATA_DIR = "/home/cip/medtech2016/eh59uqiv/data/data/simulation/highIodine"
IMAGE_LOG_DIR = "/home/cip/medtech2016/eh59uqiv/hist"
#DATA_DIR = "/home/cip/medtech2016/eh59uqiv/data/data/simulation/HIRanGeo"

# insert mean and std of trainingsrun here
# mean = [0.08751187324523926, 71.52521623883929]
# std = [71.52521623883929]
# labelsNorm = transforms.Normalize(mean=mean, std=std)
labelsNorm = None
load_params = { 'batch_size': 4,
                'shuffle': False,
                'num_workers': 5}

"""
------------------------------ params ------------------------------
"""

use_cuda = torch.cuda.is_available()
device = torch.device("cpu") # use cpu for histogram calculations


# trainset part of this dataset
ranGeo = CONRADataset(DATA_DIR,
                         True,
                         device=device,
                         precompute=True,
                         transform=labelsNorm)

loader = DataLoader(ranGeo, **load_params)

upper_limit = 0
counter = 0
number_of_bins = 100

with torch.no_grad():

    for _, mat in tqdm(loader):
        counter += 1
        mat = mat.to(device=device, dtype=torch.float)
        maxiod = max(upper_limit, np.amax((mat.cpu().numpy()[:, 0, :, :])))
        maxwater = max(upper_limit, np.amax((mat.cpu().numpy()[:, 1, :, :])))

    maxiod = int(maxiod + 0.5) #aufrunden
    maxwater = int(maxwater + 0.5)
    iod_hist = np.zeros(number_of_bins)
    water_hist = np.zeros(number_of_bins)
    print("maximum pathlengths (i/w): {} / {}".format(maxiod, maxwater))

    for _, mat in tqdm(loader):
        mat = mat.to(device=device, dtype=torch.float)
        curr_water_hist = np.histogram(np.trim_zeros(np.sort(mat.numpy()[:, 1, :, :].flatten())), bins=number_of_bins, range=[0, maxwater], density=True)
        curr_iod_hist = np.histogram(np.trim_zeros(np.sort(mat.numpy()[:, 0, :, :].flatten())), bins=number_of_bins, range=[0, maxiod], density=True)
        if np.all(np.isfinite(curr_water_hist[0])):
            water_hist += curr_water_hist[0]
        if np.all(np.isfinite(curr_iod_hist[0])):
            iod_hist += curr_iod_hist[0]


    water_axis = np.linspace(0, maxwater, number_of_bins)
    iod_axis = np.linspace(0, maxiod, number_of_bins)

    fig, (ax1, ax2) = plt.subplots(nrows=2,
                                    ncols=1)

    water_hist /= counter
    iod_hist /= counter
    ax1.plot(iod_axis, iod_hist/np.amax(iod_hist))
    ax1.title.set_text("iodine")
    ax1.set_xlabel("pathlength in mm")
    ax1.set_ylabel("relative frequency")

    # set water histogram and discriptions
    ax2.plot(water_axis, water_hist/np.amax(water_hist))
    ax2.title.set_text("water")
    ax2.set_xlabel("pathlength in mm")
    ax2.set_ylabel("relative frequency")

    # title of entire figure
    fig.suptitle("test set: relative pathlength distribution per channel")

    plt.subplots_adjust(hspace=0.5)
    plt.savefig(os.path.join(IMAGE_LOG_DIR, 'MECTPathlengthHistograms.pdf'))