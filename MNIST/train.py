from MNIST.mnistDataset import MnistDataset, ToTensor
from MNIST.models.models import ConvModel

from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn

from tqdm import trange, tqdm
from tensorboard import program
import os

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
DATA_FOLDER = 'data/MNIST/numpy'
RUN_PATH = 'run/'
useTensorboard = False
'''
--------------------------helper methods--------------------------
'''
def docu(model, testset, loss_fn):
    l = 0
    pbar = tqdm(total=len(testset))
    for batch_idx, sample in enumerate(testset):
        y_pred = model(sample['image'].view(BATCHSIZE, 1, IMAGESIZE, IMAGESIZE))
        l += loss_fn(y_pred, sample['label']).item()
        pbar.update(BATCHSIZE)
    pbar.close()
    return l
'''
---------------------------preparations---------------------------
'''
# create a folder for the weights and tensorboard logs
weigth_dir = os.path.join(RUN_PATH, "weights")
log_dir = os.path.join(RUN_PATH, "log")
if not os.path.isdir(weigth_dir):
    os.makedirs(weigth_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

# create tensorboard logger and start tensorboard
logger = None
if useTensorboard:
    logger = SummaryWriter(log_dir=log_dir)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    tb_url = tb.launch()
    print("tensorboard living on ", tb_url)

'''
-----------------------------training-----------------------------
'''
totensor = ToTensor()
traindata = MnistDataset(DATA_FOLDER, True, transform=totensor)
testdata = MnistDataset(DATA_FOLDER, False, transform=totensor)
trainingset = DataLoader(traindata, batch_size=10, shuffle=True, num_workers=5)
testset = DataLoader(testdata, batch_size=10, shuffle=False, num_workers=5)

conv_model = ConvModel()

print("there are ",  conv_model.parameters().__sizeof__(), " trainable parameters")

loss_fn = nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(conv_model.parameters(), lr=LR)

l = docu(conv_model, testset, loss_fn)
print(l)

input()

for e in trange(EPOCHS):
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
        if e * ITER + batch_idx % LOGInterval == 0:
            print("evaluating testset")
            l = docu(conv_model, testset, loss_fn)
            if useTensorboard:
                logger.add_scalar("loss", l, global_step=e * ITER + batch_idx)
        if batch_idx == ITER:
            break

logger.close()