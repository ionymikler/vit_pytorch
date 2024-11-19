# Made by: Jonathan Mikler
# Creation date: 2024-11-15
import os
import torch

import torch.nn as nn

from tqdm.auto import tqdm
from datetime import datetime
from transformers import get_scheduler

# Own
import train_utils.cifar_utils as cifar_utils
from train_utils.trainer import Trainer
from train_utils import make_optimizer, print_dict, get_args, get_cfg
from train_utils.logger_utils import create_logger, get_unique_filename

from vision_transformer import VisionTransformer

def main():
    args = get_args()
    cfg = get_cfg(args.config)
    print_dict(cfg) # TODO: improve print with logger

    _log_dir = cfg["log_dir"] if os.path.isabs(cfg["log_dir"]) else os.path.join(cfg["base_path"], cfg["log_dir"])
    logger = create_logger(name="vit_train", file_path=os.path.join(_log_dir, cfg["log_filename"]) )

    logger.info(f"loading configration from {args.config}")
    ####### Dataset setup #######
    logger.info("Setting up dataset")
    dataset_cfg = cfg["cifar_dataset"]

    label2id, id2label = cifar_utils.get_label_dicts(dataset_cfg["label_type"])

    train_dataloader, validation_dataloader, test_dataloader = cifar_utils.dataloaders_from_cfg(cfg)
    
    model = VisionTransformer(
        image_size=cfg["cifar_dataset"]["image_size"], use_linear_patch=True, num_classes=len(label2id.keys()))

    logger.info("Dataset setup complete")

    ####### Training setup #######
    # TODO: review ViT paper to implement correct lr's
    num_epochs = cfg["training"]["num_epochs"]
    lr = cfg["lr_scheduler"]["lr"]
    num_training_steps = num_epochs * len(train_dataloader)

    optimizer = make_optimizer(optimizer_name='adamw',model=model, lr=lr)
    lr_scheduler = get_scheduler(
        name=cfg["lr_scheduler"]["type"], optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    loss_function = nn.CrossEntropyLoss()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    trainer = Trainer(model,
        train_dataloader, validation_dataloader, test_dataloader,
        optimizer, lr_scheduler, loss_function, device)
    
    _final_results_dir = os.path.join(cfg["training"]["results_dir"], f"vit_{datetime.now().strftime('%H_%M_%S')}")
    _final_results_dir = _final_results_dir if os.path.isabs(_final_results_dir) else os.path.join(cfg["base_path"], _final_results_dir)
    trainer.set_cfg_params(cfg["training"])
    trainer.set_logger(logger)

    logger.info('###################  Training  ##################')
    trainer.train()

    logger.info('###################  Testing  ##################')
    trainer.test()

    logger.info("DONE")


if __name__ == '__main__':
    main()