import numpy as np
import torch
import os
import glob
from tqdm import tqdm
import argparse


def constructTorchDump(inp, output, split, dev):
    scans = [i for i in os.listdir(os.path.abspath(inp)) if os.path.isdir(os.path.join(os.path.abspath(inp), i)) and "_" in i]
    print("detected {} samples in dir \"{}\"".format(len(scans), os.path.abspath(inp)))
    # if input("continue? [y], n") == 'n':
    #     return
    samples = []
    for scan in scans:
        naming_scheme = "{}/**/*.raw".format(os.path.join(inp, scan))
        samples.append([os.path.abspath(n) for n in glob.glob(naming_scheme, recursive=True)])
    X, Y, P = detectSizesFromFilename(samples[0][0])

    # create the directories if needed
    train_output = os.path.join(output, "train")
    test_output = os.path.join(output, "test")
    split_idx = int(0.7 * len(scans) * P)
    if split:
        if not os.path.isdir(train_output):
            os.makedirs(train_output)
            output = train_output
        if not os.path.isdir(test_output):
            os.makedirs(test_output)

    CP = 2
    CM = 2
    print("precomputing dataset to: {}".format(os.path.abspath(output)))
    for i, sample in enumerate(tqdm(samples)):
        poly, mat = readToNumpy(sample, X, Y, P, CM, CP)
        for p in range(P):
            matp = mat[:, p, :, :].reshape(CM, Y, X)
            polyp = poly[:, p, :, :].reshape(CP, Y, X)
            matp = torch.tensor(matp, device=dev)
            polyp = torch.tensor(polyp, device=dev)
            # splitting the data
            sample_idx = (i * P) + p
            if sample_idx >= split_idx and split:
                sample_idx -= split_idx
                output = test_output
            torch.save(matp, os.path.join(os.path.abspath(output), "mat_{}.pt".format(sample_idx)))
            torch.save(polyp, os.path.join(os.path.abspath(output), "poly_{}.pt".format(sample_idx)))
    if split:
        return split_idx, (len(scans) * P) - split_idx
    else:
        return len(scans) * P


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
    parser.add_argument('--nosplit', action='store_false', default=True,
                        help='creates two subdirs and splits the data into them')
    args = parser.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0"

    constructTorchDump(args.in_dir, args.out, args.nosplit, device)
