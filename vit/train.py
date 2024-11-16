# Made by: Jonathan Mikler
# Creation date: 2024-11-15
import os
import yaml
import argparse
import torch
import torch.nn as nn

from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import get_scheduler

import train_utils
from vit import VisionTransformer

def get_cfg(config_path:str):
    assert (os.path.exists(config_path)), "config file path does not exists"
    return yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    
def get_args():
    parser = argparse.ArgumentParser(description='ViT')
    parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
    parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
    return parser.parse_args()

def main():
    args = get_args()
    # cfg = get_cfg(args.config)

    # TODO: Implement logger
    
    ####### Dataset setup #######
    cifar = load_dataset("uoft-cs/cifar100")
    train_label_type = "coarse" # cfg["dataset"]["train_label_type"]

    train_dataloader, test_dataloader = train_utils.make_cifar_dataloaders(cifar, batch_size=4)
    label2id_coarse, id2label_coarse = train_utils.get_cifar_label_dicts(cifar, train_label_type)

    model = VisionTransformer(
        image_size=32, use_linear_patch=True, num_classes=len(label2id_coarse.keys()))


    ####### Optimizer and lr-scheduler #######
    # TODO: review ViT paper to implement correct lr's
    num_epochs = 3 # TODO: set a param in the config file
    num_training_steps = num_epochs * len(train_dataloader)

    optimizer = train_utils.make_optimizer(optimizer_name='adamw',model=model, lr=0.003)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    loss_function = nn.CrossEntropyLoss()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["pixel_values"])

            loss = loss_function(outputs, batch[f"{train_label_type}_label"])
            loss.backward()
            train_loss += loss.item()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

if __name__ == '__main__':
    main()