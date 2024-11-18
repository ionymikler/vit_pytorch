
import os
import torch
import json
from datetime import datetime

from torch.utils.data import DataLoader
from tqdm import tqdm

# Own
from train_utils.logger_utils import copy_logger

class Trainer:
    def __init__(self,
        model:torch.nn.Module,
        train_dataloader:DataLoader, validation_dataloader:DataLoader, test_dataloader:DataLoader,
        optimizer, lr_scheduler, loss_function, device):

        self.model:torch.nn.Module = model

        self.train_dataloader:DataLoader = train_dataloader
        self.validation_dataloader:DataLoader = validation_dataloader
        self.test_dataloader:DataLoader = test_dataloader
        # TODO: Implement logic for testing strategies (epoch, batch, etc)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.device = device

    def set_cfg_params(self, cfg:dict):
        self.validation_epoch_interval:int = cfg["validation_epoch_interval"]
        self.num_epochs:int = cfg["num_epochs"]

        self.results_dir = os.path.join(cfg["results_dir"], f"vit_{datetime.now().strftime('%H_%M_%S')}")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def set_logger(self, logger):
        self.logger = copy_logger(logger, "trainer")

    def train(self, save_model=False):
        self.logger.info(f"Training for {self.num_epochs * len(self.train_dataloader)} iterations.\n \
                        Epochs: {self.num_epochs}. Batches: {len(self.train_dataloader)} of size {self.train_dataloader.batch_size}"+"\n"+"-"*30)

        _progress_bar = tqdm(range(self.num_epochs * len(self.train_dataloader)))
        best_val_loss = float('inf')
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

                _progress_bar.update(1)            
            epoch_avg_train_loss/=len(self.train_dataloader)

            # Validation
            if epoch % self.validation_epoch_interval == 0:
                self.logger.info(f'Validating at epoch {epoch}')
                epoch_avg_validation_loss, validation_accuracy = self.validate()

                self.logger.info(f'-- train loss {epoch_avg_train_loss:.3f} -- validation loss: {validation_accuracy:.3f} -- validation accuracy: {epoch_avg_validation_loss:.3f}')

                if epoch_avg_validation_loss <= best_val_loss and save_model:
                    torch.save(self.model.state_dict(), os.path.join(self.results_dir,'checkpoints/model.pth'))
                    best_val_loss = epoch_avg_validation_loss

        self.logger.info("Trainig done")
    
    def validate(self):
        self.model.eval()
        epoch_avg_val_loss = 0
        with torch.no_grad():
            total_examples, correct_predictions= 0.0, 0.0
            for batch in self.validation_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch["pixel_values"])
                
                epoch_avg_val_loss += self.loss_function(outputs, batch["coarse_label"]).item() # '.item()' retrieves only scalar
                pred_labels = outputs.argmax(dim=1)
                
                total_examples += float(len(batch['coarse_label']))
                correct_predictions += float((batch["coarse_label"] == pred_labels).sum().item())

            validation_accuracy = correct_predictions / total_examples
            epoch_avg_val_loss /= len(self.validation_dataloader)
        
        return epoch_avg_val_loss, validation_accuracy
    
    def test(self, save_results=False):
        self.logger.info(f"Testing with {len(self.train_dataloader)} batches of {self.train_dataloader.batch_size} images each"+"\n"+"-"*30)

        accuracy = 0.0
        total_examples = len(self.test_dataloader)
        self.model.eval()

        with torch.no_grad():
            batch_examples_num, batch_labels= 0.0, 0.0

            for batch in tqdm(self.test_dataloader): # QUESTION: Needed to do in batch?
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(batch["pixel_values"])
                pred_labels = outputs.argmax(dim=1)
                
                batch_examples_num += float(len(batch['coarse_label']))
                batch_labels += float((batch["coarse_label"] == pred_labels).sum().item())

                batch_acc = batch_labels / batch_examples_num
                accuracy += batch_acc
                
            accuracy /= total_examples # average accuracy

        if save_results:
            # check results_dir exists
            assert (self.results_dir is not None and os.path.exists(self.results_dir)), "results_dir not found"
            results = {
                "accuracy": accuracy,
                "total_examples": total_examples
            }
            with open(f'{self.results_dir}/test_results.json', 'w') as f:
                json.dump(results, f)

        self.logger.info(f"Test accuracy: {accuracy}")


        