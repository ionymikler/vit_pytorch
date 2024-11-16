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

class Trainer:
    def __init__(self,
        model:torch.nn.Module, train_dataloader, validation_dataloader, validation_epoch_period,
        optimizer, lr_scheduler, loss_function, num_epochs, device, logger):

        self.model:torch.nn.Module = model

        self.train_dataloader:DataLoader = train_dataloader
        self.validation_dataloader:DataLoader = validation_dataloader
        self.validation_epoch_period:int = validation_epoch_period
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.device = device
        self.num_epochs = num_epochs

        self.logger = copy_logger(logger, "trainer")

    def train(self):
        self.logger.info('###################  Training  ##################')
        progress_bar = tqdm(range(self.num_epochs * len(self.train_dataloader)))
        for epoch in range(self.num_epochs):

            # Training
            epoch_avg_train_loss = 0
            self.model.train()
            for batch in self.train_dataloader:
                # transfer batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # forward pass and loss calculation
                outputs = self.model(batch["pixel_values"])
                loss = self.loss_function(outputs, batch["coarse_label"])
                loss.backward()
                epoch_avg_train_loss += loss.item()

                # backward pass
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
            
            epoch_avg_train_loss/=len(self.train_dataloader)

            # Validation
            # TODO: Abstract validation to a method that return just accuracy and epoch_loss
            if epoch % self.validation_epoch_period == 0:
                self.logger.info(f'Validating at epoch {epoch}')
                epoch_avg_val_loss = 0
                with torch.no_grad():
                    self.model.eval()
                    total_examples, correct_predictions= 0.0, 0.0
                    for batch in self.validation_dataloader:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        outputs = self.model(batch["pixel_values"])
                        
                        epoch_avg_val_loss += self.loss_function(outputs, batch["coarse_label"]).item() # retrieve only the scalar value
                        pred_labels = outputs.argmax(dim=1)
                        
                        total_examples += float(len(batch['coarse_label']))
                        correct_predictions += float((batch["coarse_label"] == pred_labels).sum().item())

                    val_acc = correct_predictions / total_examples
                    epoch_avg_val_loss /= len(self.validation_dataloader)

                    print(
                        f'-- train loss {epoch_avg_train_loss:.3f} -- validation loss: {val_acc:.3f} -- validation accuracy: {epoch_avg_val_loss:.3f}')
                    # if epoch_avg_val_loss <= best_val_loss and save_model:
                    #     torch.save(self.model.state_dict(), 'model.pth')
                    #     best_val_loss = epoch_avg_val_loss

        self.logger.info("Trainig done")

def get_cfg(config_path:str):
    assert (os.path.exists(config_path)), "config file path does not exists"
    return yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    
def get_args():
    parser = argparse.ArgumentParser(description='ViT')
    parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
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