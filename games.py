import numpy as np
import os
import glob

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
    f = "/home/cip/medtech2016/eh59uqiv/data/run/overfitting/additionalLOGS"
    t = "/home/cip/medtech2016/eh59uqiv/data/run/overfitting/target"
    if not os.path.isdir(t):
        os.makedirs(t)

    createOverview(f, t)