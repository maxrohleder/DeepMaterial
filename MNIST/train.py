from mnistDataset import MnistDataset, ToTensor
from models.models import MNISTNet

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from tqdm import tqdm
from tensorboard import program
import os
import argparse

'''
-------------------------Hyperparameters--------------------------
'''
# Training settings
parser = argparse.ArgumentParser(description='DeepMaterial model training')
parser.add_argument('--data', default='../data/numpy',
                    help='folder containing test and training sets of MNIST')
parser.add_argument('--run', default='./run',
                    help='target folder which will hold model weights and tb logs')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--workers', type=int, default=10, metavar='N',
                    help='parallel data loading processes (default: 10)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--iterations', type=int, default=10, metavar='N',
                    help='training cycles per epoch (before validation) (default: 10)')
parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                    help='learning rate (default: 0.000001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--tb', action='store_true', default=True,
                    help='enables/disables tensorboard logging')
parser.add_argument('--seed', type=int, default=1998, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model (default=True)')

parser.add_argument('--use-tv', action='store_true', default=False,
                    help='use torchvision dataset insteat of custom one')
args = parser.parse_args()

EPOCHS = args.epochs
START = 0 # could enter a checkpoint start epoch
ITER = args.iterations  # per epoch
LR = args.lr
MOM = args.momentum
LOGInterval = args.log_interval
BATCHSIZE = args.batch_size
TEST_BATCHSIZE = args.test_batch_size
NUMBER_OF_WORKERS = args.workers
IMAGESIZE = 28
DATA_FOLDER = args.data
ROOT = args.run
WEIGHT_DIR = os.path.join(ROOT, "weights")
LOG_DIR = os.path.join(ROOT, "log")
CHECKPOINT = os.path.join(WEIGHT_DIR, "checkpoint.pt")
useTensorboard = args.tb

# check existance of data
if not os.path.isdir(DATA_FOLDER):
    print("data folder not existant or in wrong layout.\n\t", DATA_FOLDER)
    exit(0)

'''
---------------------------preparations---------------------------
'''

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

# create a folder for the weights and tensorboard logs
if not os.path.isdir(WEIGHT_DIR):
    os.makedirs(WEIGHT_DIR)
if not os.path.isdir(LOG_DIR) and useTensorboard:
    os.makedirs(LOG_DIR)

# create tensorboard logger and start tensorboard
logger = None
if useTensorboard:
    logger = SummaryWriter(log_dir=LOG_DIR)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', LOG_DIR])
    tb_url = tb.launch()
    print("tensorboard living on ", tb_url)
else:
    print('tensorboard logging turned off')

'''
----------------loading model and checkpoints---------------------
'''
print("loading datasets to ram")
if not args.use_tv:
    totensor = ToTensor()
    traindata = MnistDataset(DATA_FOLDER, True, transform=totensor)
    testdata = MnistDataset(DATA_FOLDER, False, transform=totensor)
else:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    traindata = datasets.MNIST(DATA_FOLDER+'mnist_train', train=True, download=True, transform=transform)
    testdata = datasets.MNIST(DATA_FOLDER+'mnist_test', train=False, download=True, transform=transform)
trainingset = DataLoader(traindata, **train_params)
testset = DataLoader(testdata, **test_params)

# m = ConvModel().to(device)
# loss_fn = nn.MSELoss(reduction='sum')

m = MNISTNet().to(device)
loss_fn = F.nll_loss

o = torch.optim.Adam(m.parameters(), lr=LR)
test_loss = None
train_loss = None

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
# calculating initial loss
if test_loss is None or train_loss is None:
    with torch.no_grad():
        test_loss = 0
        for x, y in testset:
            x, y = x.to(device), y.to(device)
            pred = m(x)
            test_loss += loss_fn(pred, y).item()

        train_loss = 0
        for x, y in testset:
            x, y = x.to(device), y.to(device)
            pred = m(x)
            train_loss += loss_fn(pred, y).item()

# printing runtime information
print("starting training for {} epochs {} iterations each\n\t{} total".format(EPOCHS, ITER, EPOCHS*ITER))
print("\tlogging interval: {}".format(LOGInterval))
for e in range(START, START+EPOCHS):
    for i in range(ITER):
        global_step = (e * ITER) + i

        # training
        print("\ntraining gs: ", global_step)
        for x, y in tqdm(trainingset):
            x, y = x.to(device), y.to(device)
            pred = m(x) # problematic as it assumes NUM_SAMPLES/BATCHSIZE = 0
            loss = loss_fn(pred, y)
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
                'loss': train_loss},
                CHECKPOINT)
            print('\tsaved progress to: ', CHECKPOINT)
            if logger is not None and train_loss is not None:
                logger.add_scalar('test_loss', test_loss, global_step=global_step)
                logger.add_scalar('train_loss', train_loss, global_step=global_step)
                print("\ttensorboard updated")
            elif train_loss is not None:
                print("\tloss of global-step {}: {}".format(global_step, train_loss))
            elif not useTensorboard:
                print("\t(tb-logging disabled) test/train loss: {}/{} ".format(test_loss, train_loss))
            else:
                print("\tno loss accumulated yet")

    # validation
    print("\nvalidating")
    with torch.no_grad():
        test_loss = 0
        for x, y in testset:
            x, y = x.to(device), y.to(device)
            pred = m(x)
            test_loss += loss_fn(pred, y).item()

        train_loss = 0
        for x, y in trainingset:
            x, y = x.to(device), y.to(device)
            pred = m(x)
            train_loss += loss_fn(pred, y).item()


# saving final results
print("saving upon exit")
torch.save({
    'epoch': EPOCHS+START,
    'model_state_dict': m.state_dict(),
    'optimizer_state_dict': o.state_dict(),
    'loss': train_loss},
    CHECKPOINT)
print('\tsaved progress to: ', CHECKPOINT)
if logger is not None and train_loss is not None:
    logger.add_scalar('test_loss', test_loss, global_step=global_step)
    logger.add_scalar('train_loss', train_loss, global_step=global_step)

if logger is not None:
    logger.close()
