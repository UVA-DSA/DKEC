#!/bin/bash
#SBATCH -J DKEC_GT_3e-5_layer2
#SBATCH -o ../log/MIMIC3-3737/seed-3407/DKEC_GatorTron_3e-5_layer2_maxpool_linear-%A.out
#SBATCH -e ../log/MIMIC3-3737/seed-3407/DKEC_GatorTron_3e-5_layer2_maxpool_linear-%A.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:2
#SBATCH -c 16
#SBATCH -t 72:00:00

module purge
module load anaconda
source activate DKEC

export MASTER_ADDR='localhost' &&  # Set to the master node's IP address if it's not local
export MASTER_PORT=5288  &&      # Any free port can be used

srun python ../main.py ../config/DKEC_GatorTron_3e-5_layer2.json
