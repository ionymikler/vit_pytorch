import os
import yaml
import argparse
import random
import torch
import numpy

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW, Adam, SGD

# Own
from train_utils.logger_utils import copy_logger

def get_cfg(config_path:str):
    assert (os.path.exists(config_path)), "config file path does not exists"
    return yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    
def get_args():
    parser = argparse.ArgumentParser(description='ViT')
    parser.add_argument('--config', dest='config', help='confguration of training', default="/home/iony/DTU/f24/thesis/vit_pytorch/vit/config/vit_train.yml")
    parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
    return parser.parse_args()

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

def print_dict(dictionary, ident='', braces=1):
  """ Recursively prints nested dictionaries."""

  for key, value in dictionary.items():
    if isinstance(value, dict):
      print(f'{ident}{braces*"["}{key}{braces*"]"}')
      print_dict(value, ident + '  ', braces + 1)
    else:
      print(f'{ident}{key} = {value}')