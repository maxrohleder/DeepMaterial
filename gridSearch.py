from torch import nn, optim

from CONRADataset import CONRADataset
from models.convNet import simpleConvNet
from models.unet import UNet

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboard import program
import os
import argparse

# helper function
def calculate_loss(set, loss_fn, length_set, dev, model):
    '''
    calculates the mean loss per sample
    :param set: a dataloader containing a set of length_set samples
    :param loss_fn: the function which shall be used to accumulate loss
    :param length_set: number of samples in set
    :param dev: device to use for calculation ('cpu' or 'cuda:0')
    :param model: model to evaluate
    :return: loss per sample as an float
    '''
    l = 0
    with torch.no_grad():
        for x, y in tqdm(set):
            x, y = x.to(device=dev, dtype=torch.float), y.to(device=dev, dtype=torch.float)
            pred = model(x)
            l += float(loss_fn(pred, y).item())
    return l/length_set

def computeMeanStdOverDataset(datasettype, DATAFOLDER, load_params, device, transform=None):
    if datasettype == 'CONRADataset':
        # computing mean and std over trainingset
        ds = CONRADataset(DATAFOLDER,
                         True,
                         device=device,
                         precompute=True,
                         transform=transform)
        trainingset = DataLoader(ds, **load_params)

        # sticking to convention iod -> 0, water -> 1
        mean = np.zeros(2)
        std = np.zeros(2)

        counter = 0
        # iterating and summing all mean and std
        for _, y in tqdm(trainingset):
            # y in shape [b, c, y, x]
            y = y.to(device=device, dtype=torch.float)
            iod = y[:, 0, :, :]
            water = y[:, 1, :, :]
            mean[0] += torch.mean(iod)
            mean[1] += torch.mean(water)
            std[0] += torch.std(iod)
            std[1] += torch.std(water)
            counter += 1
        return mean/counter, std/counter
    print("[train.py/computeMeanStd: dataset not recognized")
    exit(1)

# main algorithm configured by argparser. see main method of this file.
def train(args):
    '''
    -------------------------Hyperparameters--------------------------
    '''
    global std, mean
    EPOCHS = args.epochs
    START = 0  # could enter a checkpoint start epoch
    ITER = args.iterations  # per epoch
    startITER = 0
    LR = args.lr
    MOM = args.momentum
    # LOGInterval = args.log_interval
    BATCHSIZE = args.batch_size
    TEST_BATCHSIZE = args.test_batch_size
    NUMBER_OF_WORKERS = args.workers
    DATA_FOLDER = args.data
    ROOT = args.run
    WEIGHT_DIR = os.path.join(ROOT, "weights")
    CUSTOM_LOG_DIR = os.path.join(ROOT, "additionalLOGS")
    CHECKPOINT = os.path.join(WEIGHT_DIR, str(args.model) + str(args.name) + ".pt")
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
    test_params = {'batch_size': TEST_BATCHSIZE,
                   'shuffle': False,
                   'num_workers': NUMBER_OF_WORKERS}

    # create a folder for the weights and custom logs
    if not os.path.isdir(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    if not os.path.isdir(CUSTOM_LOG_DIR):
        os.makedirs(CUSTOM_LOG_DIR)


    '''
    ----------------loading model and checkpoints---------------------
    '''
    labelsNorm = None
    # normalizing on a trainingset wide mean and std
    if args.norm == 'normalize':
        print('computing mean and std over trainingset')
        # computes mean and std over all ground truths in dataset to tackle the problem of numerical insignificance
        mean, std = computeMeanStdOverDataset('CONRADataset', DATA_FOLDER, train_params, device)
        print('iodine (mean/std): {}\t{}'.format(mean[0], std[0]))
        print('water (mean/std): {}\t{}'.format(mean[1], std[1]))
        labelsNorm = transforms.Normalize(mean=mean, std=std)
        m2, s2 = computeMeanStdOverDataset('CONRADataset', DATA_FOLDER, train_params, device, transform=labelsNorm)
        print("new mean and std are (m/s): {} / {}".format(m2, s2))

    # scaling only the iodine domain to tackle the problem of low iodine numbers
    if args.norm is 'iod1000':
        labelsNorm = int(1000)


    traindata = CONRADataset(DATA_FOLDER,
                             True,
                             device=device,
                             precompute=True,
                             transform=labelsNorm)
    testdata = CONRADataset(DATA_FOLDER,
                            False,
                            device=device,
                            precompute=True,
                            transform=labelsNorm)

    trainingset = DataLoader(traindata, **train_params)
    testset = DataLoader(testdata, **test_params)

    if args.model == "unet":
        m = UNet(2, 2).to(device)
    else:
        m = simpleConvNet(2, 2).to(device)

    o = optim.SGD(m.parameters(),
                  lr=LR,
                  momentum=MOM)

    loss_fn = nn.MSELoss()

    test_loss = None
    train_loss = None

    # TODO recognice latest checkpoint and load it
    if len(os.listdir(WEIGHT_DIR)) != 0:
        checkpoints = os.listdir(WEIGHT_DIR)
        checkDir = {}
        latestCheckpoint = 0
        toUse = 0
        for i, checkpoint in enumerate(checkpoints):
            stepOfCheckpoint = int(checkpoint.split(str(args.model) + str(args.name))[-1].split('.pt')[0])
            checkDir[stepOfCheckpoint] = checkpoint
            latestCheckpoint = max(latestCheckpoint, stepOfCheckpoint)
            print("[{}] {}".format(stepOfCheckpoint, checkpoint))
        # if on development machine, prompt for input, else just take the most recent one
        if 'faui' in os.uname()[1]:
            toUse = int(input("select checkpoint to use: "))
        else:
            toUse = latestCheckpoint
        checkpoint = torch.load(os.path.join(WEIGHT_DIR, checkDir[toUse]))
        m.load_state_dict(checkpoint['model_state_dict'])
        m.to(device) # pushing weights to gpu
        o.load_state_dict(checkpoint['optimizer_state_dict'])
        train_loss = checkpoint['train_loss']
        test_loss = checkpoint['test_loss']
        START = checkpoint['epoch']
        print("using checkpoint {}:\n\tloss(train/test): {}/{}".format(toUse, train_loss, test_loss))
    else:
        print("starting from scratch")

    '''
    -----------------------------training-----------------------------
    '''
    global_step = 0
    # calculating initial loss
    if test_loss is None or train_loss is None:
        print("calculating initial loss")
        m.eval()
        print("testset...")
        test_loss = calculate_loss(set=testset, loss_fn=loss_fn, length_set=len(testdata), dev=device, model=m)
        print("trainset...")
        train_loss = calculate_loss(set=trainingset, loss_fn=loss_fn, length_set=len(traindata), dev=device, model=m)


    # printing runtime information
    print("starting training at {} for {} epochs {} iterations each\n\t{} total".format(START, EPOCHS, ITER, EPOCHS * ITER))

    print("\tbatchsize: {}\n\tloss: {}\n\twill save results to \"{}\"".format(BATCHSIZE, train_loss, CHECKPOINT))
    print("\tmodel: {}\n\tlearningrate: {}\n\tmomentum: {}\n\tnorming output space: {}".format(args.model, LR, MOM, args.norm))

    #start actual training loops
    for e in range(START, START + EPOCHS):
        # iterations will not be interupted with validation and metrics
        for i in range(startITER, ITER):
            global_step = (e * ITER) + i

            # training
            m.train()
            iteration_loss = 0
            for x, y in tqdm(trainingset):
                x, y = x.to(device=device, dtype=torch.float), y.to(device=device, dtype=torch.float)
                pred = m(x)
                loss = loss_fn(pred, y)
                iteration_loss += loss.item()
                o.zero_grad()
                loss.backward()
                o.step()
            print("\niteration {}: --accumulated loss {}".format(global_step, iteration_loss))

        # validation, saving and logging
        print("\nvalidating")
        m.eval() # disable dropout batchnorm etc
        print("testset...")
        test_loss = calculate_loss(set=testset, loss_fn=loss_fn, length_set=len(testdata), dev=device, model=m)
        print("trainset...")
        train_loss = calculate_loss(set=trainingset, loss_fn=loss_fn, length_set=len(traindata), dev=device, model=m)

        print("advanced metrics")
        #TODO come up with some metrics to evaluate image quality
        with torch.no_grad():
            for x, y in testset:
                # x, y in shape[2,2,480,620] [b,c,h,w]
                x, y = x.to(device=device, dtype=torch.float), y.to(device=device, dtype=torch.float)
                pred = m(x)
                iod = pred.cpu().numpy()[0, 0, :, :]
                water = pred.cpu().numpy()[0, 1, :, :]
                gtiod = y.cpu().numpy()[0, 0, :, :]
                gtwater = y.cpu().numpy()[0, 1, :, :]
                if(args.norm is "normalize"):
                    iod = (iod * std[0]) + mean[0]
                    water = (water * std[1]) + mean[1]
                    gtiod = (gtiod * std[0]) + mean[0]
                    gtwater = (gtwater * std[1]) + mean[1]
                if(args.norm is "iod1000"):
                    iod /= 1000
                    gtiod /= 1000
                IMAGE_LOG_DIR = os.path.join(CUSTOM_LOG_DIR, str(global_step))
                if not os.path.isdir(IMAGE_LOG_DIR):
                    os.makedirs(IMAGE_LOG_DIR)

                plt.imsave(os.path.join(IMAGE_LOG_DIR, 'iod' + str(global_step) + '.png'), iod, cmap='gray')
                plt.imsave(os.path.join(IMAGE_LOG_DIR, 'water' + str(global_step) + '.png'), water, cmap='gray')
                plt.imsave(os.path.join(IMAGE_LOG_DIR, 'gtiod' + str(global_step) + '.png'), gtiod, cmap='gray')
                plt.imsave(os.path.join(IMAGE_LOG_DIR, 'gtwater' + str(global_step) + '.png'), gtwater, cmap='gray')

                print("creating and saving profile plot at 240")
                fig2, (ax1, ax2) = plt.subplots(nrows=2,
                                                ncols=1)  # plot water and iodine in one plot
                ax1.plot(iod[240])
                ax1.plot(gtiod[240])
                ax1.title.set_text("iodine horizontal profile")
                ax1.set_ylabel("mm iodine")
                ax1.set_ylim([np.min(gtiod), np.max(gtiod)])
                print("max value in gtiod is {}".format(np.max(gtiod)))
                ax2.plot(water[240])
                ax2.plot(gtwater[240])
                ax2.title.set_text("water horizontal profile")
                ax2.set_ylabel("mm water")
                ax2.set_ylim([np.min(gtwater), np.max(gtwater)])

                plt.subplots_adjust(wspace=0.3)
                plt.savefig(os.path.join(IMAGE_LOG_DIR, 'ProfilePlots' + str(global_step) + '.png'))

                # np.save(os.path.join(IMAGE_LOG_DIR, 'iod' + str(global_step) + '.npy'), iod)
                # np.save(os.path.join(IMAGE_LOG_DIR, 'water' + str(global_step) + '.npy'), water)f
                # np.save(os.path.join(IMAGE_LOG_DIR, 'gtiod' + str(global_step) + '.npy'), gtiod)
                # np.save(os.path.join(IMAGE_LOG_DIR, 'gtwater' + str(global_step) + '.npy'), gtwater)
                break
        print("saved truth and prediction in shape " + str(iod.shape))
        print("logging")
        if logger is not None and train_loss is not None:
            logger.add_scalar('test_loss', test_loss, global_step=global_step)
            logger.add_scalar('train_loss', train_loss, global_step=global_step)
            logger.add_image("iodine-prediction", iod.reshape(1, 480, 620), global_step=global_step)
            logger.add_image("ground-truth", gtiod.reshape(1, 480, 620), global_step=global_step)
            # logger.add_image("water-prediction", wat)
            print("\ttensorboard updated with test/train loss and a sample image")
        elif train_loss is not None:
            print("\tloss of global-step {}: {}".format(global_step, train_loss))
        elif not useTensorboard:
            print("\t(tb-logging disabled) test/train loss: {}/{} ".format(test_loss, train_loss))
        else:
            print("\tno loss accumulated yet")

    # saving final results
    print("saving upon exit")
    torch.save({
        'epoch': EPOCHS,
        'iterations': ITER,
        'model_state_dict': m.state_dict(),
        'optimizer_state_dict': o.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss},
        CHECKPOINT)
    print('\tsaved progress to: ', CHECKPOINT)
    if logger is not None and train_loss is not None:
        logger.add_scalar('test_loss', test_loss, global_step=global_step)
        logger.add_scalar('train_loss', train_loss, global_step=global_step)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='DeepMaterial model training')
    parser.add_argument('--data', '-d', required=True,
                        help='folder containing test and training sets of MNIST')
    parser.add_argument('--run', '-r', required=True,
                        help='target folder which will hold model weights and logs')
    parser.add_argument('--model', '-m', required=True,
                        help='model to use. options are: [<unet>], <conv>')
    parser.add_argument('--epochs', type=int, required=True,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--iterations', type=int, required=True,
                        help='training cycles per epoch (before validation) (default: 1)')
    parser.add_argument('--lr_FROM', type=float, required=True,
                        help='learning rate (default: 0.000001)')
    parser.add_argument('--lr_TO', type=float, required=True,
                        help='learning rate (default: 0.000001)')
    parser.add_argument('--lr_STEPS', type=int, required=True,
                        help='learning rate (default: 5)')

    parser.add_argument('--workers', type=int, default=10, metavar='N',
                        help='parallel data loading processes (default: 5)')
    parser.add_argument('--name', default='checkpoint',
                        help='naming of checkpoint saved')
    parser.add_argument('--norm', required=False, default='normalize',
                        help='choose to normalize or convert iodine images to um. <normalize>, <iod1000>, <subtractmean>')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--test-batch-size', type=int, default=2, metavar='N',
                        help='input batch size for testing (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--tb', action='store_false', default=True,
                        help='enables/disables tensorboard logging')

    args = parser.parse_args()
    # handle tensorboard outside of train() to be able to stop the tensorboard process
    TB_DIR = os.path.join(args.run, "tblog")
    if not os.path.isdir(TB_DIR) and args.tb:
        os.makedirs(TB_DIR)

    # create tensorboard logger and start tensorboard
    logger = None
    if args.tb:
        logger = SummaryWriter(log_dir=TB_DIR)
    else:
        print('tensorboard logging turned off')

    try:
        lr_range = np.arange(float(args.lr_FROM), float(args.lr_TO), (float(args.lr_TO)-float(args.lr_FROM))/float(args.lr_STEPS))
        root = args.run
        for l in lr_range:
            new_root = os.path.join(root, str(l))
            if not os.path.isdir(new_root):
                os.makedirs(new_root)
            args.run = new_root
            args.lr = l
            #train(args)
        if logger is not None:
            logger.close()
    except (KeyboardInterrupt, SystemExit):
        print("exiting safely because of Keyboard Interrupt or SystemExit")
        if logger is not None:
            logger.close()