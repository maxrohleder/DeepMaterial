import pyconrad.autoinit
import time
import numpy as np
import torch
from torchvision import transforms
from CONRADataset import CONRADataset
from torch.utils.data import DataLoader

BATCHSIZE = 3
NUMBER_OF_WORKERS = 5
DATA_FOLDER = "/home/mr/Documents/bachelor/data/simulation"

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print("using device: ", str(device))

# Dataloader Parameters
train_params = {'batch_size': BATCHSIZE,
                'shuffle': True,
                'num_workers': NUMBER_OF_WORKERS}
test_params = {'batch_size': BATCHSIZE,
               'shuffle': False,
               'num_workers': NUMBER_OF_WORKERS}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

traindata = CONRADataset(DATA_FOLDER,
                         True,
                         device=device,
                         precompute=True,
                         transform=normalize)
testdata = CONRADataset(DATA_FOLDER,
                        False,
                        device=device,
                        precompute=True,
                        transform=normalize)

trainingset = DataLoader(traindata, **train_params)
testset = DataLoader(testdata, **test_params)

for i, (x, y) in enumerate(trainingset):
    print("x: {}, y: {}".format(x.size(), y.size()))


# file = '/home/mr/Documents/bachelor/data/simulation/0731125340_0/iodine/MAT_620x480x200.raw'
#
# dt = np.dtype('>f4')
# f = np.fromfile(file=file, dtype=dt)
# f = f.reshape(200, 480, 620)
#
# print(f.shape, f.dtype)
# pyconrad.imshow(f, 'poly120')
#
# f_le = f.newbyteorder().byteswap()
# pyconrad.imshow(f_le, 'poly120le')
#
#
# print(f_le.shape, f_le.dtype)
#
# input()
#
# pyconrad.close_all_windows()
# pyconrad.terminate_pyconrad()

# print('Enjoy white noise!')
# for i in range(300):
#     noise = np.random.rand(200, 200)
#     pyconrad.imshow(noise, 'White noise', spacing=[200, 2], origin=[0, 2])
#     time.sleep(0.01)
#
# _ = pyconrad.ClassGetter()
#
# array = np.random.rand(4, 2, 3).astype(pyconrad.java_float_dtype)
# grid = _.NumericGrid.from_numpy(array)
#
#
#
# pyconrad.ij()
# pyconrad.close_all_windows()
# #pyconrad.terminate_pyconrad()