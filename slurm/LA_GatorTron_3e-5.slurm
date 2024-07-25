#!/bin/bash
#SBATCH -J LA_GT
#SBATCH -o ../log/MIMIC3-3737/seed-3407/LA_GatorTron_3e-5-%A.out
#SBATCH -e ../log/MIMIC3-3737/seed-3407/LA_GatorTron_3e-5-%A.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH -t 72:00:00

module purge
module load anaconda
source activate EMSProject

export MASTER_ADDR='localhost' &&  # Set to the master node's IP address if it's not local
export MASTER_PORT=8878  &&      # Any free port can be used

srun python ../main.py ../config/LA_GatorTron_3e-5.json
