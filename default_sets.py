from collections import defaultdict
import random
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import numpy as np
import time
import json
import itertools
import torch.multiprocessing
from config import Config
from logger import is_main_process
import sys
torch.multiprocessing.set_sharing_strategy('file_system')



if is_main_process():
    with open(sys.argv[1], 'r') as fin:
        cfg = json.load(fin)
    print(cfg)
config = Config(config_file=sys.argv[1])

# with open('config/DKEC_CNN_1e-4.json', 'r') as f:
#     conf = json.load(f)
# config = Config(conf)

try:
    note = config.train.note
except:
    note = ''

if config.model == 'DKEC':
    date = config.model + '_' + config.train.backbone + '_' + config.dataset + '_' + str(config.train.graph_layer) + '_' + note
elif config.model == 'BERT':
    date = config.model + '_' + config.train.backbone + '_' + config.dataset + '_' + note
elif config.model == 'BERT_LA':
    date = config.model + '_' + config.train.backbone + '_' + config.dataset + '_' + note
else:
    date = config.model + '_' + config.dataset

num_train_epochs = config.train.epochs
dataset = config.dataset
task = config.task
ROOT = config.root_dir
DIR = os.path.join(ROOT, 'dataset', dataset)



if dataset == "RAA":
    assert config.train.window_size == 1 or config.train.window_size == None
    fname = '%s/EMS_Protocol.json' % DIR
    with open(fname, 'r') as f:
        label = json.load(f)

    with open(os.path.join(DIR, 'hier2p.json'), 'r') as f:
        hier2label = json.load(f)

    with open(os.path.join(DIR, 'p2hier.json'), 'r') as f:
        label2hier = json.load(f)

elif "MIMIC3" in dataset:
    fname = '%s/ICD9_descriptions.json' % DIR
    with open(fname, 'r') as f:
        ICD9_description = json.load(f)

    fname = '%s/ICD9CODES.json' % DIR
    with open(fname, 'r') as f:
        ICD9_DIAG = json.load(f)
    label = list(ICD9_DIAG.keys())

    fname = os.path.join(DIR, 'hier2p.json')
    with open(fname, 'r') as f:
        hier2label = json.load(f)

    fname = os.path.join(DIR, 'p2hier.json')
    with open(fname, 'r') as f:
        label2hier = json.load(f)

else:
    raise Exception('check the dataset in config')



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic\
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True