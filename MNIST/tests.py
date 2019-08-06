from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from models.models import MNISTNet
import torch

CHECKPOINT = "run/weights/checkpoint.pt"

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist_train', train=False, download=True, transform=transform)
model = MNISTNet()
c = torch.load(CHECKPOINT)
model.load_state_dict(c['model_state_dict'])

i = 0
while(True):
    x, y = trainset[i]
    plt.imshow(x.numpy().reshape(28, 28))
    _, indices = model(x).max(1)
    plt.title("truth: " + str(y.item()) + " pred: " + str(indices.item()))
    plt.show()
    ja = input("show another example? [y]/n")
    if ja == 'y' or ja == '':
        i += 1
        continue
    break
