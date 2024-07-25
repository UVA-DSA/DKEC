from tqdm import tqdm
import torch
from default_sets import *
import wandb
from utils import coarseMap
import json

def train_fn(epoch, data_loader, model, optimizer, scheduler, graph, loss_fn):
    '''
        Modified from Abhishek Thakur's BERT example:
        https://github.com/abhishekkrthakur/bert-sentiment/blob/master/src/engine.py
    '''

    train_loss = 0.0
    model.train()
    model.zero_grad()
    progress_bar = tqdm(
        data_loader,
        desc='Epoch {:1d}/{:1d}'.format(epoch, num_train_epochs))

    for batch in progress_bar:
        ids = batch["ids"]
        mask = batch["mask"]
        targets = batch["labels"].to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids, mask, graph, target=targets)

        if isinstance(outputs, tuple):
            outputs, loss = outputs[0], outputs[1]
        else:
            loss = loss_fn(outputs, targets, is_multi=True)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        train_loss += loss.item()

        optimizer.step()
        scheduler.step()

    return train_loss


def eval_fn(data_loader, G, model, loss_func):
    '''
        Modified from Abhishek Thakur's BERT example:
        https://github.com/abhishekkrthakur/bert-sentiment/blob/master/src/engine.py
    '''
    eval_loss = 0.0
    model.eval()
    fin_targets = []
    fin_outputs = []
    fin_ages = []
    with torch.no_grad():
        for bi, d in enumerate(tqdm(data_loader, desc='eval')):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["labels"].to(device, dtype=torch.float)
            ages = d['age']
            fin_ages.extend(ages)

            outputs = model(ids=ids, mask=mask, G=G, target=targets)

            if isinstance(outputs, tuple):
                outputs, loss = outputs[0], outputs[1]

            if loss_func != None:
                loss = loss_func(outputs, targets, is_multi=True)
                eval_loss += loss.item()

            fin_targets.extend(targets)

            if task == 'multi_class':
                _, preds = torch.max(outputs, dim=1)
                fin_outputs.extend(preds)
            elif task == 'multi_label':
                fin_outputs.extend(torch.sigmoid(outputs))
            else:
                raise Exception('check task in default_sets.py')

    return eval_loss, fin_outputs, fin_targets, fin_ages