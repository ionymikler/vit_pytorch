# Made by: Jonathan Mikler
# Creation date: 2024-11-15
import os
import yaml
import argparse
import torch
import torch.nn as nn
import logging 

from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import get_scheduler

# Own
from train_utils import make_optimizer
from train_utils import Trainer, cifar_utils, get_args, get_cfg
from train_utils.logger_utils import create_logger, get_unique_filename

from vision_transformer import VisionTransformer

def main():
    args = get_args()
    # cfg = get_cfg(args.config)
    # logger = create_logger(cfg["log_filepath"])
    logger = create_logger(get_unique_filename("./logs/vit_train.log"), "vit_train")

    ####### Dataset setup #######
    logger.info("Loading dataset")
    batch_size = 4
    cifar = load_dataset("uoft-cs/cifar100")
    train_label_type = "coarse" # cfg["dataset"]["train_label_type"]

    train_dataloader, test_dataloader = cifar_utils.dataloader_from_dataset_dict(cifar, batch_size=batch_size)
    label2id_coarse, id2label_coarse = cifar_utils.get_cifar_label_dicts(cifar, train_label_type)

    model = VisionTransformer(
        image_size=32, use_linear_patch=True, num_classes=len(label2id_coarse.keys()))

    ####### Optimizer and lr-scheduler #######
    # TODO: review ViT paper to implement correct lr's
    num_epochs = 3 # TODO: set a param in the config file
    lr = 0.003
    num_training_steps = num_epochs * len(train_dataloader)

    optimizer = make_optimizer(optimizer_name='adamw',model=model, lr=0.003)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    loss_function = nn.CrossEntropyLoss()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    trainer = Trainer(model, train_dataloader, optimizer, lr_scheduler, loss_function, device, logger, num_epochs)
    trainer.train()

    logger.info("DONE")


if __name__ == '__main__':
    main()