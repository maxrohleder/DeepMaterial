from MNIST.mnistDataset import MnistDataset, ToTensor
from MNIST.models.models import ConvModel

from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

import torch

'''
-------------------------Hyperparameters--------------------------
'''
EPOCHS = 50
ITER = 100
LR = 1e-6
MOM = 0.5
LOGInterval = 100
BATCHSIZE = 10
IMAGESIZE = 28
KERNELNUMBER_C1 = 10
DATA_FOLDER = 'data/numpy'
'''
--------------------------helper methods--------------------------
'''


def docu(model, testset, loss_fn):
    l = 0
    for batch_idx, sample in enumerate(testset):
        y_pred = model(sample['image'].view(BATCHSIZE, 1, IMAGESIZE, IMAGESIZE))
        l += loss_fn(y_pred, sample['label']).item()
    return l

'''
-----------------------------training-----------------------------
'''
totensor = ToTensor()
traindata = MnistDataset(DATA_FOLDER, True, transform=totensor)
testdata = MnistDataset(DATA_FOLDER, False, transform=totensor)
trainingset = DataLoader(traindata, batch_size=10, shuffle=True, num_workers=5)
testset = DataLoader(testdata, batch_size=10, shuffle=True, num_workers=5)

conv_model = ConvModel()

loss_fn = nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(conv_model.parameters(), lr=LR)

loss = 0
loss_decrease = 1
loss_list = [0] * 5


l = docu(conv_model, testset, loss_fn)
print(l)

input()

pbar = tqdm(range(EPOCHS))

while loss_decrease > 0.4:
    for e in pbar:
        for batch_idx, sample in enumerate(trainingset):
            # forward pass
            y_pred = conv_model(sample['image'].view(BATCHSIZE, 1, IMAGESIZE, IMAGESIZE))
            # loss evaluation
            loss = loss_fn(y_pred, sample['label'])
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # updating weigths
            optimizer.step()
            # logging
            if (e+1) * (batch_idx+1) % LOGInterval == 0:
                print("evaluating testset")
                l = docu(conv_model, testset, loss_fn)
                print(l)
                loss_decrease = loss_list.pop(0) - l
                loss_list.append(loss.item())

            if batch_idx == ITER:
                break

pbar.close()
