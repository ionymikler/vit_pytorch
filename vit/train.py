# Made by: Jonathan Mikler
# Creation date: 2024-11-15
import torch
import torch.nn as nn

from tqdm.auto import tqdm
from datasets import load_dataset, DatasetDict, Dataset
from transformers import get_scheduler

# Own
from train_utils import make_optimizer, print_dict
from train_utils import Trainer, cifar_utils, get_args, get_cfg
from train_utils.logger_utils import create_logger, get_unique_filename

from vision_transformer import VisionTransformer

def main():
    args = get_args()
    cfg = get_cfg(args.config)
    print_dict(cfg) # TODO: improve print with logger

    unique_logfile_path = get_unique_filename(cfg["log_filepath"])
    logger = create_logger(unique_logfile_path, "vit_train")

    logger.info(f"all logs saved in {unique_logfile_path}")

    ####### Dataset setup #######
    logger.info("Setting up dataset")
    batch_size = cfg["training"]["batch_size"]

    cifar_train:Dataset = load_dataset("uoft-cs/cifar100", split="train[:100]")
    cifar_validation:Dataset = load_dataset("uoft-cs/cifar100", split="test[:10]")

    train_label_type = cfg["cifar_dataset"]["train_label_type"]

    train_dataloader = cifar_utils.dataloader_from_dataset(cifar_train, batch_size=batch_size)
    validation_dataloader = cifar_utils.dataloader_from_dataset(cifar_validation, batch_size=batch_size)
    label2id_coarse, id2label_coarse = cifar_utils.get_cifar_label_dicts(cifar_train, train_label_type)

    model = VisionTransformer(
        image_size=cfg["cifar_dataset"]["image_size"], use_linear_patch=True, num_classes=len(label2id_coarse.keys()))

    ####### Optimizer and lr-scheduler #######
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

    trainer = Trainer(
        model,
        train_dataloader, validation_dataloader, cfg["training"]["validation_epoch_interval"],
        optimizer, lr_scheduler, loss_function, num_epochs, device=device, logger=logger)
    trainer.train()

    logger.info("DONE")


if __name__ == '__main__':
    main()