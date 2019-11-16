import numpy as np
import os
import glob
import torch

def mkifnot(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def createOverview(additionalLOGS, targetfolder):
    '''
    converts the advanced metrics of train.py to raw files readable by imagej
    :param additionalLOGS:
    :param targetfolder:
    :return:
    '''
    # output is raw byteformat 32 bit real, big endian byte order naming: x_y_z
    # input is numpy dataformat np.float32 used by torch

    filepattern = additionalLOGS + '/**/*.npy'
    print("found {} files matching the following pattern:".format(len(os.listdir(additionalLOGS))))
    print(filepattern)
    input("continue?")

    # target folders for the new raw files
    tiod = targetfolder + '/iod'
    twater = targetfolder + '/water'
    tgtiod = targetfolder + '/gtiod'
    tgtwater = targetfolder + '/gtwater'
    mkifnot(tiod)
    mkifnot(twater)
    mkifnot(tgtiod)
    mkifnot(tgtwater)

    # loop over all files in log, convert and sort them
    for npyFile in glob.iglob(filepattern, recursive=True):
        _, fleext = os.path.split(npyFile)
        fle, ext = os.path.splitext(fleext)
        folder = ""
        # sort
        if 'iod' in fle:
            folder = tiod
            if 'gtiod' in fle:
                folder = tgtiod
        if 'water' in fle:
            folder = twater
            if 'gtwater' in fle:
                folder = tgtwater

        img = np.load(npyFile).astype('>f4')
        newFle = os.path.join(folder, fle + '_{}x{}.raw'.format(img.shape[0], img.shape[1]))
        img.tofile(newFle)





if __name__ == "__main__":
    # f = "/Users/mr/bachelor/data/run/overf_MECT_iodOnly/additionalLOGS"
    # t = "/Users/mr/bachelor/data/run/overf_MECT_iodOnly/results"
    # if not os.path.isdir(t):
    #     os.makedirs(t)
    #
    # createOverview(f, t)
    a = np.zeros((2, 2, 5, 5))
    t = torch.from_numpy(a)
    t.data[:, 0, :, :] *= 1
    t.data[]
