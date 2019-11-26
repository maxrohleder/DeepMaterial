from torch import nn, optim

from CONRADataset import CONRADataset
from models.convNet import simpleConvNet
from models.unet import UNet

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision import transforms

import numpy as np
from skimage.measure import compare_ssim as ssim
from scipy.stats import pearsonr
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
                # norming iodine data range as its below one and results in np.nan
                maxIod = iodine.max()
                if maxIod != 0:
                    iodFlatNormed = (iodine.flatten()/maxIod)*100
                    gtiodFlatNormed = (gti.flatten()/maxIod)*100
                    iodR += pearsonr(iodFlatNormed, gtiodFlatNormed)[0] / 200

                iodSSIM += ssim(iodine, gti) / 200
                waterR += pearsonr(water.flatten(), gtw.flatten())[0] / 200
                waterSSIM += ssim(water, gtw) / 200

    return [iodSSIM, waterSSIM], [iodR, waterR]



# main algorithm configured by argparser. see main method of this file.
def evaluate_performance(args, gridargs, logger):
    '''
    -------------------------Hyperparameters--------------------------
    '''
    EPOCHS = args.epochs
    ITER = args.iterations  # per epoch
    LR = gridargs['lr']
    MOM = gridargs['mom']
    # LOGInterval = args.log_interval
    BATCHSIZE = args.batch_size
    NUMBER_OF_WORKERS = args.workers
    DATA_FOLDER = args.data
    ROOT = gridargs['run']
    CUSTOM_LOG_DIR = os.path.join(ROOT, "additionalLOGS")

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

    '''
    ---------------------------loading dataset and normalizing---------------------------
    '''
    # Dataloader Parameters
    train_params = {'batch_size': BATCHSIZE,
                    'shuffle': True,
                    'num_workers': NUMBER_OF_WORKERS}
    test_params = {'batch_size': BATCHSIZE,
                   'shuffle': False,
                   'num_workers': NUMBER_OF_WORKERS}

    # create a folder for the weights and custom logs
    if not os.path.isdir(CUSTOM_LOG_DIR):
        os.makedirs(CUSTOM_LOG_DIR)

    traindata = CONRADataset(DATA_FOLDER,
                             True,
                             device=device,
                             precompute=True,
                             transform=None)

    testdata = CONRADataset(DATA_FOLDER,
                            False,
                            device=device,
                            precompute=True,
                            transform=None)

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
    print("starting training at {} for {} epochs {} iterations each\n\t{} total".format(0, EPOCHS, ITER, EPOCHS * ITER))

    print("\tbatchsize: {}\n\tloss: {}\n".format(BATCHSIZE, train_loss))
    print("\tmodel: {}\n\tlearningrate: {}\n\tmomentum: {}\n\tnorming output space: {}".format(args.model, LR, MOM, False))

    #start actual training loops
    for e in range(0, EPOCHS):
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

        print("calculating performace...")
        currSSIM, currR = performance(set=testset, dev=device, model=m, bs=BATCHSIZE)
        print("SSIM (iod/water): {}/{}\nR (iod/water): {}/{}".format(currSSIM[0], currSSIM[1], currR[0], currR[1]))
        #f.write("num, lr, mom, step, ssimIOD, ssimWAT, rIOD, rWAT, trainLOSS, testLOSS\n")
        with open(gridargs['stats'], 'a') as f:
            newCSVline = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(gridargs['runnum'], LR,
                                                                           MOM, global_step,
                                                                           currSSIM[0], currSSIM[1],
                                                                           currR[0],    currR[1],
                                                                           train_loss,  test_loss)
            f.write(newCSVline)
            print("wrote new line to csv:\n\t{}".format(newCSVline))

        print("advanced metrics")
        with torch.no_grad():
            for x, y in testset:
                # x, y in shape[2,2,480,620] [b,c,h,w]
                x, y = x.to(device=device, dtype=torch.float), y.to(device=device, dtype=torch.float)
                pred = m(x)
                iod = pred.cpu().numpy()[0, 0, :, :]
                water = pred.cpu().numpy()[0, 1, :, :]
                gtiod = y.cpu().numpy()[0, 0, :, :]
                gtwater = y.cpu().numpy()[0, 1, :, :]

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
                break

        if logger is not None and train_loss is not None:
            logger.add_scalar('test_loss', test_loss, global_step=global_step)
            logger.add_scalar('train_loss', train_loss, global_step=global_step)
            logger.add_image("iodine-prediction", iod.reshape(1, 480, 620), global_step=global_step)
            logger.add_image("ground-truth", gtiod.reshape(1, 480, 620), global_step=global_step)
            # logger.add_image("water-prediction", wat)
            print("\ttensorboard updated with test/train loss and a sample image")

    # saving final results
    CHECKPOINT = os.path.join(args.run, "finalWeights.pt")
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

    parser.add_argument('--model', '-m', default='unet',
                        help='model to use. options are: [<unet>], <conv>')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--workers', type=int, default=10, metavar='N',
                        help='parallel data loading processes (default: 5)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--iterations', type=int, default=1, metavar='N',
                        help='training cycles per epoch (before validation) (default: 1)')

    parser.add_argument('--lrfrom', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.000001)')
    parser.add_argument('--lrto', type=float, default=1e-6, metavar='LR',
                        help='learning rate (default: 0.000001)')

    parser.add_argument('--momfrom', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--momto', type=float, default=1.1, metavar='M',
                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--steps', type=int, default=5,
                        help='steps in lr and mom direction')

    # general runtime args
    args = parser.parse_args()
    # additional args needed for gridsearch
    gridargs = {}

    # logfile to later contain all runtime infos of all runs
    stats = os.path.join(args.run, "stats.csv")
    gridargs['stats'] = stats
    # folder to save all individial logs of all runs t0
    runs = os.path.join(args.run, "runs")


    if not os.path.isdir(args.run):
        os.makedirs(args.run)
    if not os.path.isdir(runs):
        os.makedirs(runs)

    with open(stats, 'w+') as f:
        f.write("num, lr, mom, step, ssimIOD, ssimWAT, rIOD, rWAT, trainLOSS, testLOSS\n")

    # space to perform gridsearch over
    lrrange = np.linspace(args.lrfrom, args.lrto, args.steps)
    momrange = np.linspace(args.momfrom, args.momto, args.steps)
    print("learningrate range: {}\nmomentum range: {}".format(lrrange, momrange))

    print("---------------------- starting hyperparamer optimization ----------------------")

    logger = None
    try:
        for i, lr in enumerate(lrrange):
            for j, mom in enumerate(momrange):
                rundir = os.path.join(runs, "lr{}mom{}".format(i, j))
                if not os.path.isdir(rundir):
                    os.makedirs(rundir)
                    ## start training with right args object
                    gridargs['runnum'] = (i  * args.steps) + j
                    gridargs['run'] = rundir
                    gridargs['lr'] = lr
                    gridargs['mom'] = mom
                    print("starting run number: {}\n\tlr: {}\n\tmom: {}".format(gridargs['runnum'], gridargs['lr'], gridargs['mom']))
                    logger = SummaryWriter(log_dir=rundir)
                    print("tensorboard logs are written to ", rundir)

                    evaluate_performance(args, gridargs, logger)

                    ## end training
                    logger.close()
    except (KeyboardInterrupt, SystemExit):
        print("exiting safely because of Keyboard Interrupt or SystemExit")
        if logger is not None:
            logger.close()