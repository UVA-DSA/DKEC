#!/bin/bash
#SBATCH -J disBERT
#SBATCH -o ../log/MIMIC3-3737/seed-3407/DistilBioBERT-%A.out
#SBATCH -e ../log/MIMIC3-3737/seed-3407/DistilBioBERT-%A.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH -t 72:00:00

module purge
module load anaconda
source activate DKEC

export MASTER_ADDR='localhost' &&  # Set to the master node's IP address if it's not local
export MASTER_PORT=44627  &&      # Any free port can be used

srun python ../main.py ../config/DistilBioBERT.json
