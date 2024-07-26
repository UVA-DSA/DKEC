import wandb
import os
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def setup_wandb(config):
    wandb.login()
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    if not (config.wandb.enable):
        return

    run = wandb.init(
        config=config,
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=f"{config.model}_{config.train.backbone}_lr:{config.train.learning_rate}_bs:{config.train.batch_size}_gl:{config.train.graph_layer}",
        reinit=True
    )
    return run


def log_dict_to_wandb(log, train_loss, val_loss):
    wandb.log({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'micro f1 avg': log['micro avg']['f1-score'],
        'macro f1 avg': log['macro avg']['f1-score'],
        'P@1': log['P@1'],
        'R@1': log['R@1'],
        'RP@1': log['R-Precision@1'],
        'nDCG@1': log['nDCG@1'],
        'P@6': log['P@6'],
        'R@6': log['R@6'],
        'RP@6': log['R-Precision@6'],
        'nDCG@6': log['nDCG@6'],
        'P@8': log['P@8'],
        'R@8': log['R@8'],
        'RP@8': log['R-Precision@8'],
        'nDCG@8': log['nDCG@8'],
        'P@12': log['P@12'],
        'R@12': log['R@12'],
        'RP@12': log['R-Precision@12'],
        'nDCG@12': log['nDCG@12'],
    })

