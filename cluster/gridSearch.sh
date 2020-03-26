#!/bin/bash
#SBATCH --job-name=grid
#SBATCH --ntasks=1
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o /home/%u/%x-%j-on-%N.out
#SBATCH -e /home/%u/%x-%j-on-%N.err
#SBATCH --mail-type=ALL
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=23:59:59

# Hyperparameters and directories
LOGDIR='/cluster/<user>/logs/gridSearch/'
DATADIR='/cluster/<user>/data/simulation/highIodine'

BS='4'
EPOCHS='50'		# will log this often
ITER='5'		# number of training cycles before logging
MODEL='unet' 	# options are 'unet', 'conv'

# configuring the slurm job
#SBATCH --job-name=gridSearch
#SBATCH --ntasks=1
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o home/%u/slurm-%j.out
#SBATCH -e home/%u/slurm-%j.err

# Tell pipenv to install the virtualenvs in the cluster folder

export PATH=/cluster/<user>/miniconda/bin:$PATH

# using existing env DeepMaterial
conda init bash
source /home/<user>/.bashrc
conda activate DeepMaterial

python /cluster/<user>/bachelor/DeepMaterial/gridSearch.py -r $LOGDIR -d $DATADIR -m $MODEL --epochs $EPOCHS --iterations $ITER --batch-size $BS

# copy this file to the log dir
cp /home/<user>/gridSearch.sh $LOGDIR

# to see termination directly in homedirectory
touch /home/<user>/GRIDSEARCHdone

exit 0