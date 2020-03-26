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
LOGDIR='/cluster/<user>/logs/grid/'
DATADIR='/cluster/<user>/data/simulation/highIodine/'

BS='4'
EPOCHS='20'		# will log this often
ITER='4'		# number of training cycles before logging
MODEL='unet' 	# options are 'unet', 'conv'

# use custom miniconda 
export PATH=/cluster/<user>/miniconda/bin:$PATH

# using existing env DeepMaterial
conda init bash
source /home/<user>/.bashrc
conda activate DeepMaterial

python /cluster/<user>/bachelor/DeepMaterial/gridSearch.py -r $LOGDIR -d $DATADIR -m $MODEL --epochs $EPOCHS --iterations $ITER --batch-size $BS


N="$(ls -l $LOGDIR | grep ^d | wc -l)"
TARGET=25

echo "there are ${N} folders in ${LOGDIR}"

while [ $TARGET -gt $N ]
do
        echo "restarting grid search with ${N} folders"
        python /cluster/<user>/bachelor/DeepMaterial/gridSearch.py -r $LOGDIR -d $DATADIR -m $MODEL --epochs $EPOCHS --iterations $ITER --batch-size $BS

        echo $N
        N="$(ls -l $LOGDIR | grep ^d | wc -l)"
        sleep 3
done
# copy this file to the log dir
cp /home/<user>/gridSearch.sh $LOGDIR

# to see termination directly in homedirectory
touch /home/<user>/GRIDSEARCHdone

exit 0