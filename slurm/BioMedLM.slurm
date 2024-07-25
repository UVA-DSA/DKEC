#!/bin/bash
#SBATCH -J BLM_1e-5
#SBATCH -o ../log/MIMIC3-3737/seed-3407/BioMedLM_1e-5-%A.out
#SBATCH -e ../log/MIMIC3-3737/seed-3407/BioMedLM_1e-5-%A.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100_80gb
#SBATCH -c 16
#SBATCH -t 72:00:00

module purge
module load anaconda
source activate EMSProject

export MASTER_ADDR='localhost' &&  # Set to the master node's IP address if it's not local
export MASTER_PORT=9067  &&      # Any free port can be used

srun python ../main.py ../config/BioMedLM.json
