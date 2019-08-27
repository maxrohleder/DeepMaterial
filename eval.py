import glob
import torch
import os
import numpy as np
from matplotlib import pyplot as plt
import argparse
from torch.utils.data import DataLoader


from CONRADataset import CONRADataset
from material import multi_slice_viewer
from models.convNet import simpleConvNet
from models.unet import UNet


def eval(poly, model, d):
    '''

    :param poly: numpy array in shape 2, p, y, x
    :param model: must be on device d and take float data in shape 2, y, x
    :param d: device to run evaluation on
    :return: numpy array material image
    '''
    print(poly.shape)
    mat = np.zeros_like(poly)
    (c, p, y, x) = poly.shape
    with torch.no_grad():
        for proj in range(poly.shape[0]):
            mat[:, proj, :, :] = model(torch.from_numpy(poly[:, proj, :, :].reshape(1, 2, y, x)).to(device=d, dtype=torch.float)).numpy()
    print(mat.shape)
    return mat


def detectSizesFromFilename(filename):
    pa, filename = os.path.split(filename)
    fle, ext = os.path.splitext(filename)
    parts = fle.split("x")
    X = int(parts[0].split('_')[1])
    Y = int(parts[1])
    P = int(parts[2].split('.')[0])
    return X, Y, P

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
        print(file + " on channel " + str(mat_channel))
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

def norm(c):
    '''
    :param c: x, y, c
    :return: channel normed version
    '''
    _ = plt.hist(c[:,:,0])
    plt.title("water histogramm")
    plt.show()
    _ = plt.hist(c[:,:,1])
    plt.title("iodine histogramm")
    plt.show()
    mean = np.sum(np.sum(c, axis=0), axis=0)/(c.shape[0]*c.shape[1])
    std = np.sqrt(np.sum(np.sum(np.square(np.subtract(c, mean)), axis=0), axis=0))/(c.shape[0]*c.shape[1])
    normed = (c-mean)/std
    normed = (normed + normed.min())
    normed = normed / normed.max()
    c[:, :, 0] = 0
    return c

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model evaluation utility')
    parser.add_argument('--data', default="../data/simulation",
                        help='folder containing data in .pt or .raw to be used for evaluation')
    parser.add_argument('--weigths', default="MNIST/run/weights/checkpoint.pt",
                        help='a checkpoint of weights for that type of model')
    parser.add_argument('--torch', default="NotSet",
                        help='a checkpoint of weights for that type of model')
    parser.add_argument('--model', '-m', default='unet',
                        help='model to use. options are: <unet>, [<simpleconv>]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use gpu')
    parser.add_argument('--interactive', '-i', dest='i', action='store_true', default=False,
                        help='show examples')

    args = parser.parse_args()
    root_dir = args.data
    CHECKPOINT = args.weigths
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda and args.cuda else "cpu")

    m = None
    if args.model == "unet":
        m = UNet(2, 2).to(device)
    else:
        m = simpleConvNet(2, 2).to(device)

    scans = [os.path.join(root_dir, i) for i in os.listdir(os.path.abspath(root_dir)) if
             os.path.isdir(os.path.join(os.path.abspath(root_dir), i)) and "_" in i]
    if len(scans) == 0:
        print("no scan data found (folder name must be in format mmddhhmmss_x with x beeing the serialnumber")
        exit()

    if not (os.path.exists(CHECKPOINT) and os.path.isfile(CHECKPOINT)):
        print("weights in wrong format or nonexistant")
        exit()

    print("loading old checkpoint from \"{}\"".format(CHECKPOINT))
    checkpoint = torch.load(CHECKPOINT)
    m.load_state_dict(checkpoint['model_state_dict'])
    m.to(device)  # pushing weights to gpu
    train_loss = checkpoint['train_loss']
    test_loss = checkpoint['test_loss']
    START = checkpoint['epoch']

    train_params = {'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 1}

    evaldata = CONRADataset(root_dir,
                             True,
                             device=device,
                             precompute=True,
                             transform=None)

    evalset = DataLoader(evaldata, **train_params)

    with torch.no_grad():
        for p, mat in evalset:
            p, mat = p.to(device=device, dtype=torch.float), mat.to(device=device, dtype=torch.float)
            pred = m(p)
            print("success")

            poly = np.vstack((p.numpy()[0], np.zeros((1, 480, 620)))).transpose((1,2,0))
            materials = np.vstack((mat.numpy()[0], np.zeros((1, 480, 620)))).transpose((1,2,0))
            predictions = np.vstack((pred.numpy()[0], np.zeros((1, 480, 620)))).transpose((1,2,0))


            # poly120 = np.vstack((p.numpy()[0, 0, :, :], np.zeros((1, 480, 620)))).transpose((1,2,0))
            # poly80 = np.vstack((p.numpy()[0, 1, :, :], np.zeros((1, 480, 620)))).transpose((1,2,0))
            # iod = np.vstack((mat.numpy()[0, 1, :, :], np.zeros((1, 480, 620)))).transpose((1,2,0))
            # water = np.vstack((mat.numpy()[0, 0, :, :], np.zeros((1, 480, 620)))).transpose((1,2,0))
            # prediod = np.vstack((pred.numpy()[0, 1, :, :], np.zeros((1, 480, 620)))).transpose((1,2,0))
            # predwater = np.vstack((pred.numpy()[0, 0, :, :], np.zeros((1, 480, 620)))).transpose((1,2,0))



            plt.imshow(poly)
            plt.title("poly")
            plt.show()
            plt.imshow(materials)
            plt.title("materials")
            plt.show()
            plt.imshow(predictions)
            plt.title("predictions")
            plt.show()
            exit()



    # for s in scans:
    #     print(s)
    #     files = glob.glob(os.path.abspath(s) + "/**/*.raw")
    #     x, y, p = detectSizesFromFilename(files[0])
    #     poly, mat = readToNumpy(files, x, y, p, 2, 2)
    #     print("shape of mat is " + str(mat.shape))
    #     print("max value is " + str(poly.max()))
    #
    #     #pred = eval(poly, m, device)
    #     _ = plt.hist(mat.flatten(), bins='auto')
    #     plt.title("material histogram")
    #     plt.show()
    #     #print("max value of prediction is " + str(pred.max()))
    #
    #     iodine = multi_slice_viewer(mat[0], "poly80 prediction")
    #     # water = multi_slice_viewer(pred[1], "water prediction")
    #     # tiod = multi_slice_viewer(mat[0], "iodine ground truth")
    #     # twater = multi_slice_viewer(mat[1], "water ground truth")
    #     iodine.show()
    #     # water.show()
    #     # tiod.show()
    #     # twater.show()
    #
    #     yn = None
    #     while(True):
    #         yn = input("show next scan? [y], n")
    #         if yn == 'y' or yn == "":
    #             yn = False
    #             break
    #         elif yn == 'n':
    #             yn = True
    #             break
    #         else:
    #             print('not understood. either enter, y or n')
    #
    #     if yn:
    #         break
