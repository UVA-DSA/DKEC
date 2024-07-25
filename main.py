from trainer import *
import torch.multiprocessing as mp
from config import Config
import wandb
import os
import default_sets
from default_sets import *
import re
import warnings
from vocab import WordVocab
import sys
import yaml
from argparse import ArgumentParser
from utils import AttrDict
warnings.filterwarnings("ignore")

def main(rank, world_size):
    print('set up ddp')
    ddp_setup(rank, world_size)
    print("Cuda support:", torch.cuda.is_available(), f"rank/devices: {rank}/{world_size}:")
    seed_everything(config.train.seed)
    save_root = {
        "result": os.path.join(ROOT, 'results/{}/train/lr:{}_bs:{}/seed:{}/reports'.format(date,
                                                                                           config.train.learning_rate,
                                                                                           config.train.batch_size,
                                                                                           config.train.seed)),
        "model": os.path.join(ROOT, 'models/{}/lr:{}_bs:{}/seed:{}'.format(date, config.train.learning_rate,
                                                                           config.train.batch_size,
                                                                           config.train.seed))
    }
    if is_main_process():
        for k, v in save_root.items():
            if not os.path.exists(v):
                os.makedirs(v)

    data, model, graph, optimizer, scheduler, loss_func = load_train_objs(config, dataset, rank, world_size)
    trainer = Trainer(config, model, graph, data, optimizer, scheduler, loss_func, rank, 1, save_root)

    if not config.test.is_test:
        if is_main_process() and config.wandb.enable:
            run = setup_wandb(config)
            url = run.get_url()
            print(f"training details: {url}")
            wandb.watch(model)

        trainer.train(config)
        if is_main_process() and config.wandb.enable:
            run.finish()

    else:
        assert config.test.is_test == True, 'test mode is incorrect'
        print("start testing...")
        trainer.test(config)

    destroy_process_group()


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '3338'
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)




