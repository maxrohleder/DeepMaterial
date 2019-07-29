import time

from MNIST.mnistDataset import MnistDataset, ToTensor
from MNIST.models.models import ConvModel

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms

from tqdm import tqdm
from tensorboard import program
import os

'''
-------------------------Hyperparameters--------------------------
'''
EPOCHS = 100
START = 0 # could enter a checkpoint start epoch
ITER = 10  # per epoch
LR = 1e-6
MOM = 0.5
LOGInterval = 50
BATCHSIZE = 50
NUMBER_OF_WORKERS = 10  # max 12
IMAGESIZE = 28
DATA_FOLDER = 'data/numpy'
ROOT = 'run/'
WEIGHT_DIR = os.path.join(ROOT, "weights")
LOG_DIR = os.path.join(ROOT, "log")
CHECKPOINT = os.path.join(WEIGHT_DIR, "checkpoint.tar")
useTensorboard = True

'''
---------------------------preparations---------------------------
'''

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Dataloader Parameters
train_params = {'batch_size': BATCHSIZE,
                'shuffle': True,
                'num_workers': NUMBER_OF_WORKERS}
test_params = {'batch_size': BATCHSIZE,
               'shuffle': False,
               'num_workers': NUMBER_OF_WORKERS}

# create a folder for the weights and tensorboard logs
if not os.path.isdir(WEIGHT_DIR):
    os.makedirs(WEIGHT_DIR)
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)

# create tensorboard logger and start tensorboard
logger = None
if useTensorboard:
    logger = SummaryWriter(log_dir=LOG_DIR)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', LOG_DIR])
    tb_url = tb.launch()
    print("tensorboard living on ", tb_url)

'''
----------------loading model and checkpoints---------------------
'''
# totensor = ToTensor()
# traindata = MnistDataset(DATA_FOLDER, True, transform=totensor)
# testdata = MnistDataset(DATA_FOLDER, False, transform=totensor)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
traindata = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
testdata = datasets.MNIST('mnist_test', train=False, download=True, transform=transform)
trainingset = DataLoader(traindata, **train_params)
testset = DataLoader(testdata, **test_params)

m = ConvModel(IMAGESIZE).to(device)
loss_fn = nn.MSELoss(reduction='sum')
o = torch.optim.Adam(m.parameters(), lr=LR)
l = None

if os.path.exists(CHECKPOINT) and os.path.isfile(CHECKPOINT):
    print("loading old checkpoint...")
    checkpoint = torch.load(CHECKPOINT)
    m.load_state_dict(checkpoint['model_state_dict'])
    m.to(device)
    o.load_state_dict(checkpoint['optimizer_state_dict'])
    l = checkpoint['loss']
    START = checkpoint['epoch']
    print("resuming training at epoch ", START, " at a loss of ", l)
else:
    print("starting from scratch")

'''
-----------------------------training-----------------------------
'''
global_step = 0
for e in range(START, START+EPOCHS):
    for i in range(ITER):
        global_step = (e * ITER) + i

        # training
        print("\ntraining gs: ", global_step)
        for x, y in tqdm(trainingset): # change this to entire trainingsset later
            x, y = x.to(device), y.to(device)
            pred = m(x).view(BATCHSIZE)
            loss = loss_fn(pred, y.to(torch.float))
            o.zero_grad()
            loss.backward()
            o.step()

        # logging if applicable
        if global_step % LOGInterval == 0:
            print("logging")
            torch.save({
                'epoch': e,
                'model_state_dict': m.state_dict(),
                'optimizer_state_dict': o.state_dict(),
                'loss': l},
                CHECKPOINT)
            print('\tsaved progress to: ', CHECKPOINT)
            if logger is not None and l is not None:
                logger.add_scalar('loss', l, global_step=global_step)
                print("\ttensorboard updated")
            elif l is not None:
                print("\tloss of global-step {}: {}".format(global_step, l))
            elif not useTensorboard:
                print("\t(tb-logging disabled)")
            else:
                print("\tno loss accumulated yet")

    # validation
    print("\nvalidating")  # tqdm needs time to chill
    with torch.no_grad():
        l = 0
        for x, y in testset:
            x, y = x.to(device), y.to(device)
            pred = m(x).view(BATCHSIZE)
            l += loss_fn(pred, y.to(torch.float)).item()


if logger is not None:
    logger.close()
