#!/bin/bash
#SBATCH -J DKEC_CNN_1e-4
#SBATCH -o ../log/MIMIC3-3737/seed-3407/DKEC_CNN_1e-4_woHier-%A.out
#SBATCH -e ../log/MIMIC3-3737/seed-3407/DKEC_CNN_1e-4_woHier-%A.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH -t 72:00:00

module purge
module load anaconda
source activate DKEC

export MASTER_ADDR='localhost' &&  # Set to the master node's IP address if it's not local
export MASTER_PORT=4466  &&      # Any free port can be used

srun python ../main.py ../config/DKEC_CNN_woHier_1e-4.json