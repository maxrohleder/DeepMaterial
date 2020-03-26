#!/bin/bash
#SBATCH --job-name=strain
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o /home/%u/slurm-%j.out
#SBATCH -e /home/%u/slurm-%j.err
#SBATCH --mail-type=ALL

# add conda to the path
export PATH=/cluster/<user>/miniconda/bin:$PATH

# using existing env DeepMaterial
conda init bash
source /home/<user>/.bashrc
conda activate DeepMaterial

python /cluster/<user>/bachelor/DeepMaterial/train.py --epochs 100 --iterations 10 --batch-size 4 -r /cluster/<user>/logs/overfitting/MECThighIodine2 -d /cluster/<user>/data/simulation/highIodine/ -m unet --lr 0.00001
