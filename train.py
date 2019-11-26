from skimage.measure import compare_ssim as ssim
from scipy.stats import pearsonr

from CONRADataset import CONRADataset
from models.convNet import simpleConvNet
from models.unet import UNet

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse

if 'faui' in os.uname()[1]:
    from tensorboard import program

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
    # NORMLABEL
    if datasettype == 'CONRADataset':
        # computing mean and std over trainingset
        ds = CONRADataset(DATAFOLDER,
                         True,
                         device=device,
                         precompute=True,
                         transform=transform)
        trainingset = DataLoader(ds, **load_params)

        # sticking to convention iod -> 0, water -> 1
        m = np.zeros(2)
        s = np.zeros(2)

        counter = 0
        # iterating and summing all mean and std
        for _, y in tqdm(trainingset):
            # y in shape [b, c, y, x]
            y = y.to(device=device, dtype=torch.float)
            iod = y[:, 0, :, :]
            water = y[:, 1, :, :]
            m[0] += torch.mean(iod)
            m[1] += torch.mean(water)
            s[0] += torch.std(iod)
            s[1] += torch.std(water)
            counter += 1
        return m/counter, s/counter
    print("[train.py/computeMeanStd: dataset not recognized")
    exit(1)

def performance(set, dev, model, bs):
    iodSSIM = 0
    waterSSIM = 0
    iodR = 0
    waterR = 0

    with torch.no_grad():
        for x, y in set:
            x, y = x.to(device=dev, dtype=torch.float), y.to(device=dev, dtype=torch.float)
            # shape is (bs, 2, 480, 620)
            pred = model(x)
            # loop over samples in batch
            for p in range(bs):
                iodine = pred[p, 0, :, :].cpu().numpy()
                water = pred[p, 1, :, :].cpu().numpy()
                gti = y[p, 0, :, :].cpu().numpy()
                gtw = y[p, 1, :, :].cpu().numpy()
                assert len(gti.shape) == 2

                iodR += pearsonr(iodine.flatten(), gti.flatten())[0] / 200
                iodSSIM += ssim(iodine, gti) / 200
                waterR += pearsonr(water.flatten(), gtw.flatten())[0] / 200
                waterSSIM += ssim(water, gtw) / 200

    return [iodSSIM, waterSSIM], [iodR, waterR]


def advanvedMetrics(groundTruth, pred, mean, std, global_step, norm, IMAGE_LOG_DIR):
    '''
    logging advanced metrics in IMAGE_LOG_DIR
    in case of stddev normalization mean will be [0, 0]
    '''
    iod = pred[0]
    water = pred[1]
    gtiod = groundTruth[0]
    gtwater = groundTruth[1]
    if norm:
        # NORMLABEL
        print('denormalizing images')
        iod = (iod * std[0]) + mean[0]
        water = (water * std[1]) + mean[1]
        gtiod = (gtiod * std[0]) + mean[0]
        gtwater = (gtwater * std[1]) + mean[1]

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

    plt.subplots_adjust(hspace=0.3)
    plt.savefig(os.path.join(IMAGE_LOG_DIR, 'ProfilePlots' + str(global_step) + '.png'))
    print("saved truth and prediction in shape " + str(iod.shape))


# main algorithm configured by argparser. see main method of this file.
def train(args):
    '''
    -------------------------Hyperparameters--------------------------
    '''
    EPOCHS = args.epochs
    START = 0  # could enter a checkpoint start epoch
    ITER = args.iterations  # per epoch
    LR = args.lr
    MOM = args.momentum
    # LOGInterval = args.log_interval
    BATCHSIZE = args.batch_size
    TEST_BATCHSIZE = args.test_batch_size
    NUMBER_OF_WORKERS = args.workers
    DATA_FOLDER = args.data
    TESTSET_FOLDER = args.testset
    ROOT = args.run
    WEIGHT_DIR = os.path.join(ROOT, "weights")
    CUSTOM_LOG_DIR = os.path.join(ROOT, "additionalLOGS")
    CHECKPOINT = os.path.join(WEIGHT_DIR, str(args.model) + str(args.name) + ".pt")
    useTensorboard = args.tb

    # check existance of data
    if not os.path.isdir(DATA_FOLDER):
        print("data folder not existant or in wrong layout.\n\t", DATA_FOLDER)
        exit(0)
    # check existance of testset
    if TESTSET_FOLDER is not None and not os.path.isdir(TESTSET_FOLDER):
        print("testset folder not existant or in wrong layout.\n\t", DATA_FOLDER)
        exit(0)


    '''
    ---------------------------preparations---------------------------
    '''

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("using device: ", str(device))

    # loading the validation samples to make online evaluations
    path_to_valX = args.valX
    path_to_valY = args.valY
    valX = None
    valY = None
    if path_to_valX is not None and path_to_valY is not None \
            and os.path.exists(path_to_valX) and os.path.exists(path_to_valY) \
            and os.path.isfile(path_to_valX) and os.path.isfile(path_to_valY):
        with torch.no_grad():
            valX, valY = torch.load(path_to_valX, map_location='cpu'), \
                   torch.load(path_to_valY, map_location='cpu')


    '''
    ---------------------------loading dataset and normalizing---------------------------
    '''
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

    labelsNorm = None
    # NORMLABEL
    # normalizing on a trainingset wide mean and std
    mean = None
    std = None
    if args.norm:
        print('computing mean and std over trainingset')
        # computes mean and std over all ground truths in dataset to tackle the problem of numerical insignificance
        mean, std = computeMeanStdOverDataset('CONRADataset', DATA_FOLDER, train_params, device)
        print('\niodine (mean/std): {}\t{}'.format(mean[0], std[0]))
        print('water (mean/std): {}\t{}\n'.format(mean[1], std[1]))
        labelsNorm = transforms.Normalize(mean=[0, 0], std=std)
        m2, s2 = computeMeanStdOverDataset('CONRADataset', DATA_FOLDER, train_params, device, transform=labelsNorm)
        print("new mean and std are:")
        print('\nnew iodine (mean/std): {}\t{}'.format(m2[0], s2[0]))
        print('new water (mean/std): {}\t{}\n'.format(m2[1], s2[1]))

    traindata = CONRADataset(DATA_FOLDER,
                             True,
                             device=device,
                             precompute=True,
                             transform=labelsNorm)

    testdata = None
    if TESTSET_FOLDER is not None:
        testdata = CONRADataset(TESTSET_FOLDER,
                                False,
                                device=device,
                                precompute=True,
                                transform=labelsNorm)
    else:
        testdata = CONRADataset(DATA_FOLDER,
                                False,
                                device=device,
                                precompute=True,
                                transform=labelsNorm)

    trainingset = DataLoader(traindata, **train_params)
    testset = DataLoader(testdata, **test_params)

    '''
    ----------------loading model and checkpoints---------------------
    '''

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

    if len(os.listdir(WEIGHT_DIR)) != 0:
        checkpoints = os.listdir(WEIGHT_DIR)
        checkDir = {}
        latestCheckpoint = 0
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

    ## SSIM and R value
    R = []
    SSIM = []
    performanceFLE = os.path.join(CUSTOM_LOG_DIR, "performance.csv")
    with open(performanceFLE, 'w+') as f:
        f.write("step, SSIMiodine, SSIMwater, Riodine, Rwater, train_loss, test_loss\n")
    print("computing ssim and r coefficents to: {}".format(performanceFLE))


    # printing runtime information
    print("starting training at {} for {} epochs {} iterations each\n\t{} total".format(START, EPOCHS, ITER, EPOCHS * ITER))

    print("\tbatchsize: {}\n\tloss: {}\n\twill save results to \"{}\"".format(BATCHSIZE, train_loss, CHECKPOINT))
    print("\tmodel: {}\n\tlearningrate: {}\n\tmomentum: {}\n\tnorming output space: {}".format(args.model, LR, MOM, args.norm))

    #start actual training loops
    for e in range(START, START + EPOCHS):
        # iterations will not be interupted with validation and metrics
        for i in range(ITER):
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

        print("calculating SSIM and R coefficients")
        currSSIM, currR = performance(set=testset, dev=device, model=m, bs=BATCHSIZE)
        print("SSIM (iod/water): {}/{}\nR (iod/water): {}/{}".format(currSSIM[0], currSSIM[1], currR[0], currR[1]))
        with open(performanceFLE, 'a') as f:
            newCSVline = "{}, {}, {}, {}, {}, {}, {}\n".format(global_step, currSSIM[0], currSSIM[1], currR[0],
                                                               currR[1], train_loss, test_loss)
            f.write(newCSVline)
            print("wrote new line to csv:\n\t{}".format(newCSVline))

        '''
            if valX and valY were set in preparations, use them to perform analytics.
            if not, use the first sample from the testset to perform analytics
        '''
        with torch.no_grad():
            truth, pred = None, None
            IMAGE_LOG_DIR = os.path.join(CUSTOM_LOG_DIR, str(global_step))
            if not os.path.isdir(IMAGE_LOG_DIR):
                os.makedirs(IMAGE_LOG_DIR)

            if valX is not None and valY is not None:
                batched = np.zeros((BATCHSIZE, *valX.numpy().shape))
                batched[0] = valX.numpy()
                batched = torch.from_numpy(batched).to(device=device, dtype=torch.float)
                pred = m(batched)
                pred = pred.cpu().numpy()[0]
                truth = valY.numpy() # still on cpu

                assert pred.shape == truth.shape
            else:
                for x, y in testset:
                    # x, y in shape[2,2,480,620] [b,c,h,w]
                    x, y = x.to(device=device, dtype=torch.float), y.to(device=device, dtype=torch.float)
                    pred = m(x)
                    pred = pred.cpu().numpy()[0] # taking only the first sample of batch
                    truth = y.cpu().numpy()[0] # first projection for evaluation
            advanvedMetrics(truth, pred, mean, std, global_step, args.norm, IMAGE_LOG_DIR)

        print("logging")
        CHECKPOINT = os.path.join(WEIGHT_DIR, str(args.model) + str(args.name) + str(global_step) + ".pt")
        torch.save({
            'epoch': e+1, # end of this epoch; so resume at next.
            'model_state_dict': m.state_dict(),
            'optimizer_state_dict': o.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss},
            CHECKPOINT)
        print('\tsaved weigths to: ', CHECKPOINT)
        if logger is not None and train_loss is not None:
            logger.add_scalar('test_loss', test_loss, global_step=global_step)
            logger.add_scalar('train_loss', train_loss, global_step=global_step)
            logger.add_image("iodine-prediction", pred[0].reshape(1, 480, 620), global_step=global_step)
            logger.add_image("water-prediction", pred[1].reshape(1, 480, 620), global_step=global_step)
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
    parser.add_argument('--valX', required=False, default=None,
                        help='path to a single .pt file to validate every epoch')
    parser.add_argument('--valY', required=False, default=None,
                        help='path to a single .pt file to validate every epoch')
    parser.add_argument('--testset', required=False, default=None,
                        help="path to dataset to use to evaluate as testset")

    parser.add_argument('--model', '-m', default='unet',
                        help='model to use. options are: [<unet>], <conv>')
    parser.add_argument('--name', default='checkpoint',
                        help='naming of checkpoint saved')
    parser.add_argument('--norm', required=False, action='store_true', default=False,
                        help='choose to normalize or convert iodine images to um. <normalize>, <iod1000>, <subtractmean>')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--test-batch-size', type=int, default=2, metavar='N',
                        help='input batch size for testing (default: 2)')
    parser.add_argument('--workers', type=int, default=10, metavar='N',
                        help='parallel data loading processes (default: 5)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--iterations', type=int, default=1, metavar='N',
                        help='training cycles per epoch (before validation) (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                        help='learning rate (default: 0.000001)')
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
        #tb = program.TensorBoard()
        #tb.configure(argv=[None, '--logdir', TB_DIR])
        #tb_url = tb.launch()
        #print("tensorboard living on ", tb_url)
        print("tensorboard logs are written to ", TB_DIR)
    else:
        print('tensorboard logging turned off')
    try:
        train(args)
        if logger is not None:
            logger.close()
    except (KeyboardInterrupt, SystemExit):
        print("exiting safely because of Keyboard Interrupt or SystemExit")
        if logger is not None:
            logger.close()