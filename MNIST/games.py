import torch
from MNIST.mnistDataset import MnistDataset, ToTensor
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from MNIST.models.models import ConvModel
from torch.nn import MSELoss
from os import path
import time

RUN_PATH = 'run/'
WEIGHTS_FILE = RUN_PATH + "weights/checkpoint.tar"
DATA = "data/numpy"
ITER = 1
START = 0
END = 5
LR = 1e-6
BATCHSIZE = 10
IMAGESIZE = 28

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

test_params = {'batch_size': BATCHSIZE,
               'shuffle': False,
               'num_workers': 10}
totensor = ToTensor()
testdata = MnistDataset(DATA, False, transform=totensor)
testset = DataLoader(testdata, **test_params)

m = ConvModel(IMAGESIZE).to(device)
loss_fn = MSELoss()
o = torch.optim.Adam(m.parameters(), lr=LR)
l = None
if path.exists(WEIGHTS_FILE) and path.isfile(WEIGHTS_FILE):
    print("loading old checkpoint...")
    checkpoint = torch.load(WEIGHTS_FILE)
    m.load_state_dict(checkpoint['model_state_dict'])
    m.to(device)
    o.load_state_dict(checkpoint['optimizer_state_dict'])
    l = checkpoint['loss']
    START = checkpoint['epoch']
    print("resuming training at epoch ", START, " at a loss of ", l)

print("number of batches: ", len(testset))
print("training on device: ", device)
time.sleep(0.5)

for e in range(START, END):
    for i in range(ITER):
        # training
        print("\ntraining")
        for x, y in tqdm(testset):
            x, y = x.to(device), y.to(device)
            pred = m(x).view(BATCHSIZE)
            loss = loss_fn(pred, y)
            o.zero_grad()
            loss.backward()
            o.step()

        # validation
        time.sleep(0.1)
        print("\nvalidating")  # tqdm needs time to chill
        with torch.no_grad():
            l = 0
            for x, y in tqdm(testset):
                x, y = x.to(device), y.to(device)
                pred = m(x).view(BATCHSIZE)
                l += loss_fn(y.view(BATCHSIZE), pred).item()
            print("\nloss of epoch {}: {}".format(e, l))

    torch.save({
        'epoch': e,
        'model_state_dict': m.state_dict(),
        'optimizer_state_dict': o.state_dict(),
        'loss': l},
        WEIGHTS_FILE)
