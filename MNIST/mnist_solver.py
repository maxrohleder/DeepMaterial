from MNIST.mnistDataset import MnistDataset, ToTensor
from MNIST.models import ConvModel

from torch.utils.data import DataLoader
from torch import nn
from tqdm import trange, tqdm

import torch
import numpy as np

'''
--------------------------Hyperparameters--------------------------
'''
EPOCHS = 50
ITER = 100
LR = 1e-6
MOM = 0.5
LOGInterval = 100
BATCHSIZE = 10
IMAGESIZE = 28
KERNELNUMBER_C1 = 10
'''
--------------------------Hyperparameters--------------------------
'''
totensor = ToTensor()
mnistdata = MnistDataset('data/numpy', True, transform=totensor)
dataloader = DataLoader(mnistdata, batch_size=10, shuffle=True, num_workers=5)

conv_model = ConvModel()

loss_fn = nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(conv_model.parameters(), lr=LR)

loss = 0
loss_decrease = 1
loss_list = [0] * 5

pbar = tqdm(range(EPOCHS))

while loss_decrease > 0.4:
    for e in pbar:
        for batch_idx, sample in enumerate(dataloader):
            y_pred = conv_model(sample['image'])
            loss = loss_fn(y_pred, sample['label'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if((e+1) * (batch_idx+1) % LOGInterval == 0):
                print((e+1)*(batch_idx+1), loss.item())
            if batch_idx == ITER:
                break
        loss_decrease = loss_list.pop(0) - loss.item()
        loss_list.append(loss.item())

pbar.close()