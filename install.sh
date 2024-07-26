#!/bin/bash

# Initialize Conda for the current session
source $(conda info --base)/etc/profile.d/conda.sh

# Create and activate the conda environment
conda create -n DKEC python=3.10.6 -y
conda activate DKEC

# Install pip
conda install pip -y

# Install PyTorch and related packages
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric packages
pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-2.2.0+cu118.html
pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-2.2.0+cu118.html
pip install -q torch-geometric==2.3.1

# Install other required packages
pip install transformers tqdm pandas numpy openpyxl pyyaml wandb nltk sacremoses notebook matplotlib gensim
