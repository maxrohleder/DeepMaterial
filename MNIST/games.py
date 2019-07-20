from torch.nn import MSELoss
from tqdm import tqdm
from MNIST.models.models import ConvModel
from MNIST.mnistDataset import MnistDataset, ToTensor
from torch.utils.data import DataLoader


RUN_PATH = 'run/'
DATA = "data/MNIST/numpy"
ITER = 10
EPOCHS = 5
BATCHSIZE = 10
IMAGESIZE = 28


def docu(model, testset, loss_fn):
    l = 0
    pbar = tqdm(total=len(testset))
    for batch_idx, sample in enumerate(testset):
        y_pred = model(sample['image'].view(BATCHSIZE, 1, IMAGESIZE, IMAGESIZE))
        l += loss_fn(y_pred, sample['label']).item()
        pbar.update(BATCHSIZE)
    pbar.close()
    return l


conv = ConvModel()

mnist_test = MnistDataset(DATA, True, ToTensor())
testset = DataLoader(mnist_test, batch_size=10, shuffle=False, num_workers=5)

print(mnist_test[0]['image'].size())

lms = MSELoss(reduction='sum')

print(docu(conv, testset, lms))