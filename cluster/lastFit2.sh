#!/bin/bash
#SBATCH --job-name=lastFit2
#SBATCH --ntasks=1
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o /home/%u/%j-%x-on-%N.out
#SBATCH -e /home/%u/%j-%x-on-%N.err
#SBATCH --mail-type=ALL
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=23:59:59

# Hyperparameters and directories
LOGDIR='/cluster/<user>/logs/lastFit2'
DATADIR='/cluster/<user>/data/simulation/HIRanGeo/'

TEST_DIR='/cluster/<user>/data/simulation/highIodine'

VALX='/cluster/<user>/data/simulation/highIodine/torch/test/poly_0.pt'
VALY='/cluster/<user>/data/simulation/highIodine/torch/test/mat_0.pt'

BS=4			# batchsize. 2 samples consume about 7.9 gb in gpu memory.	
LR=0.00001		# e-4
EPOCHS=100		# Epochs * Iterations = total train time
ITER=10			# this defines the logging interval. every ITER iterations logging is performed
MODEL=unet 		# options are 'unet', 'conv'

# add conda to the path
export PATH=/cluster/<user>/miniconda/bin:$PATH

# using existing env DeepMaterial
conda init bash
source /home/<user>/.bashrc
conda activate DeepMaterial

# starting the training
python /cluster/<user>/bachelor/DeepMaterial/train.py --testset $TEST_DIR --valX $VALX --valY $VALY --epochs $EPOCHS --iterations $ITER --batch-size $BS -r $LOGDIR -d $DATADIR -m $MODEL --lr $LR --norm

# copy this file to the log dir
cp /home/<user>/lastFit.sh $LOGDIR

# to see termination directly in homedirectory
touch /home/<user>/EXITSUCCESS

exit 0