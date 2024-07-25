from transformers import AdamW
from torch.optim import SGD, Adam

def ems_optimizer(model, config):
    '''
    Taken from Abhishek Thakur's Tez library example:
    https://github.com/abhishekkrthakur/tez/blob/main/examples/text_classification/binary.py
    '''
    param_optimizer = list(model.named_parameters())

    if config.model == 'CNN':
        optimizer_parameters = model.parameters()
    else:
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": config.optimizer.decay_rate,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

    if config.optimizer.optimizer_type == 'AdamW':
        opt = AdamW(optimizer_parameters, lr=config.train.learning_rate)
    elif config.optimizer.optimizer_type == 'Adam':
        opt = Adam(optimizer_parameters, lr=config.train.learning_rate)
    else:
        opt = None
    return opt