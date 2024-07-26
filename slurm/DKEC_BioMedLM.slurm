#!/bin/bash
#SBATCH -J DKEC_BLM_1e-6_layer2
#SBATCH -o ../log/MIMIC3-3737/seed-3407/DKEC_BioMedLM_1e-6_layer2-%A.out
#SBATCH -e ../log/MIMIC3-3737/seed-3407/DKEC_BioMedLM_1e-6_layer2-%A.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH -c 16
#SBATCH -t 72:00:00

module purge
module load anaconda
source activate DKEC

export MASTER_ADDR='localhost' &&  # Set to the master node's IP address if it's not local
export MASTER_PORT=44259  &&      # Any free port can be used

srun python ../main.py ../config/DKEC_BioMedLM_1e-6_layer2.json
