import random
import torch
import numpy

from torch.optim import Optimizer, AdamW, Adam, SGD

def make_optimizer(optimizer_name, model:torch.nn.Module, lr)->Optimizer:
    if 'adamw' == optimizer_name:
        optimizer = AdamW(model.parameters(), lr)
    else:
        raise NotImplementedError(f"{optimizer_name} optimizer is not supported")

    return optimizer


def set_random_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed ** 2)
    torch.manual_seed(seed ** 3)
    torch.cuda.manual_seed(seed ** 4)
    torch.cuda.manual_seed_all(seed ** 4)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True