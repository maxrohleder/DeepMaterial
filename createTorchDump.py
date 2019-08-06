import numpy as np
import torch
import os
import glob
from tqdm import tqdm
import argparse

def constructTorchDump(input, output, split, device):

    numSamples = sum(os.path.isdir(os.path.join(input, i)) and "_" in i for i in os.listdir(input))
    print("detected {} samples in dir \"{}\"".format(numSamples, os.path.abspath(input)))
    samples = []
    for i in range(numSamples):
        naming_scheme = "{}/*_{}/**/*.raw".format(input, i)
        samples.append([n for n in glob.glob(naming_scheme, recursive=True)])
    X, Y, P = detectSizesFromFilename(samples[0][0])

    # create the directories if needed
    train_output = os.path.join(output, "train")
    test_output = os.path.join(output, "test")
    split_idx = int(0.7 * numSamples)
    if split:
        if not os.path.isdir(train_output):
            os.makedirs(train_output)
        if not os.path.isdir(test_output):
            os.makedirs(test_output)

    CP = 2
    CM = len(samples[0]) - CP
    print("precomputing dataset to: {}".format(os.path.abspath(output)))
    for i, sample in enumerate(tqdm(samples)):
        poly, mat = readToNumpy(sample, X, Y, P, CM, CP)
        if i < split_idx and split:
            output = train_output
        if i >= split_idx and split:
            i -= split_idx
            output = test_output
        for p in range(P):
            matp = mat[:, p, :, :].reshape(CM, Y, X)
            polyp = poly[:, p, :, :].reshape(CP, Y, X)
            matp = torch.tensor(matp, device=device)
            polyp = torch.tensor(polyp, device=device)
            torch.save(matp, output + "/mat_{}.pt".format((i * P) + p))
            torch.save(polyp, output + "/poly_{}.pt".format((i * P) + p))

def readToNumpy(files, X, Y, P, CM, CP):
    '''
    reads in big endian 32 bit float raw files and outputs numpy data
    :param files: string array containing the paths to poly and mat files
    :return: numpy arrays in format [C, P, Y, X] C = number of materials/energy projections
    '''
    dt = np.dtype('>f4')
    poly = np.empty((CP, P, Y, X))
    mat = np.empty((CM, P, Y, X))
    mat_channel = 0
    for file in files:
        if 'POLY80' in file:
            f = np.fromfile(file=file, dtype=dt).newbyteorder().byteswap()
            poly[0] = f.reshape(P, Y, X)
        elif 'POLY120' in file:
            f = np.fromfile(file=file, dtype=dt).newbyteorder().byteswap()
            poly[1] = f.reshape(P, Y, X)
        elif 'MAT' in file:
            f = np.fromfile(file=file, dtype=dt).newbyteorder().byteswap()
            mat[mat_channel] = f.reshape(P, Y, X)
            mat_channel += 1
        else:
            print("ERROR IN FILE STRUCTURE")
    return poly, mat

def detectSizesFromFilename(filename):
    pa, filename = os.path.split(filename)
    fle, ext = os.path.splitext(filename)
    parts = fle.split("x")
    X = int(parts[0].split('_')[1])
    Y = int(parts[1])
    P = int(parts[2].split('.')[0])
    return X, Y, P

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='torch dump creation utility')
    parser.add_argument('--in', dest="in_dir", default='/home/mr/Documents/bachelor/data/simulation',
                        help='directory containing the CONRAD output')
    parser.add_argument('--out', default='.',
                        help='folder containing test and training sets of MNIST')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='loades the data to cpu not gpu')
    parser.add_argument('--split', action='store_true', default=True,
                        help='creates two subdirs and splits the data into them')
    args = parser.parse_args()
    device = "cpu" if args.cpu else "cuda:0"

    constructTorchDump(args.in_dir, args.out, args.split, device)

