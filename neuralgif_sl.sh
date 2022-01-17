#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 12:00:00
#SBATCH -o "/scratch/inf0/user/gtiwari/slurm-%A.out"
#SBATCH -e "/scratch/inf0/user/gtiwari/slurm-%A.err"
#SBATCH --gres gpu:1

echo "neuralGIF pytorch implementation"
cd /BS/garvita/work/code/neuralgif
source /BS/garvita/static00/software/miniconda3/etc/profile.d/conda.sh
conda activate if-net_10

python trainer.py