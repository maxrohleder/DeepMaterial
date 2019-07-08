import numpy as np
import gzip
import os

SIZE = 28

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
N_TRAIN = 1
TRAIN_PREFIX = 'train'

TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
N_TEST = 1
TEST_PREFIX = 'test'


def mnistToNumpy(mnistfolder, targetfolder):
    '''
    this method takes the paths to a folder containing ONLY the origial MNIST dataset (with .gz extention), converts
    them into numpy data format and saves the images, one by one into the specified folder

    :param mnistfolder: folder containing dataset from http://yann.lecun.com/exdb/mnist/
    :param targetfolder: destination folder to safe .npy files to. (will structure into train and test 6/1)
    :return: 0 on success 1 on error
    '''
    if not os.path.exists(targetfolder):
        print('MNIST data not found! aborting')
        return False
    else:
        if not os.path.exists(os.path.join(targetfolder, TRAIN_IMAGES))\
                and os.path.exists(os.path.join(targetfolder, TRAIN_LABELS))\
                and os.path.exists(os.path.join(targetfolder, TEST_IMAGES))\
                and os.path.exists(os.path.join(targetfolder, TEST_LABELS)):
            print('MNIST data not found! aborting')
            return False

    if not os.path.exists(targetfolder):
        os.mkdir(targetfolder)
        print("Directory ", targetfolder, " Created ")
    else:
        print("Directory ", targetfolder, " already exists")

    tr_img = os.path.join(mnistfolder, TRAIN_IMAGES)
    tr_lab = os.path.join(mnistfolder, TRAIN_LABELS)

    te_img = os.path.join(mnistfolder, TEST_IMAGES)
    te_lab = os.path.join(mnistfolder, TEST_LABELS)

    tr_target = os.path.join(targetfolder, 'train/')
    te_target = os.path.join(targetfolder, 'test/')

    # create training target folder
    if not os.path.exists(tr_target):
        os.mkdir(tr_target)
        print("Directory ", tr_target, " Created ")
    else:
        print("Directory ", tr_target, " already exists")

    # creating test target folder
    if not os.path.exists(te_target):
        os.mkdir(te_target)
        print("Directory ", te_target, " Created ")
    else:
        print("Directory ", te_target, " already exists")

    stat = doConversion(tr_img, tr_lab, N_TRAIN, tr_target, TRAIN_PREFIX)
    stat = stat and doConversion(te_img, te_lab, N_TEST, te_target, TEST_PREFIX)
    return stat


def doConversion(img_path, label_path, n, target_folder, prefix):

    imgs = gzip.open(img_path, 'r')
    labels = gzip.open(label_path, 'r')

    # read out header
    imgs.read(16)
    labels.read(8)

    img_buf = imgs.read(SIZE * SIZE * n)
    label_buf = labels.read(n)

    data = np.frombuffer(img_buf, dtype=np.uint8).astype(np.float32)
    print('read', prefix, 'images')
    labels = np.frombuffer(label_buf, dtype=np.uint8).astype(int)
    print('read', prefix, 'labels')
    data = data.reshape(n, SIZE, SIZE)
    print('---------- SAVING', prefix, 'DATA ----------------')

    for i in range(n):
        name = prefix + "_" + str(i) + "_" + str(labels[i]) + ".npy"
        target_file_path = os.path.join(target_folder, name)
        if not os.path.exists(target_file_path):
            np.save(target_file_path, data[i])
            print("file ", target_file_path, "  created")
        else:
            print("file ", target_file_path, " already exists")

    print('----------------- DONE ---------------------')
    return True


mnistToNumpy("data/MNIST/orig/", "data/MNIST/numpy/")
#mnistToNumpy("", "")
