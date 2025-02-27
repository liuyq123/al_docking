import argparse
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb

from utils.data.dataset import GraphIterableDataset
from utils.data.data_transformer import DataTransformer
from src.model_creator import ModelCreator
from src.scheduler_creator import SchedulerCreator
from utils.utils import yaml_parser


class Trainer:
    def __init__(self, config):
        self.project_name = config['wandb']['project']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ModelCreator(config['model']).get_model()
        self.model = self.model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.num_epoch = config['optimization']['num_epoch']
        self.optimizer = AdamW(self.model.parameters(), 
                               lr=config['optimization']['learning_rate'],
                               weight_decay=0.0001)
        self.scheduler = SchedulerCreator(config['optimization']).get_scheduler(self.optimizer)

        train_data_transformer = DataTransformer(config['data']['transformation_strategy'])
        self.train_dataset = GraphIterableDataset(config['data']['training'], 
                                                  shuffle=True, 
                                                  batch_size=config['data']['batch_size'],
                                                  data_transformer=train_data_transformer,
                                                  mode='training')
        self.valid_dataset = GraphIterableDataset(config['data']['validation'], 
                                                  shuffle=True, 
                                                  batch_size=config['data']['batch_size'],
                                                  data_transformer=train_data_transformer,
                                                  mode='training')
        
        self.smoothing_factor = config['optimization']['early_stopping']['smoothing_factor']
        self.initial_training = config['optimization']['early_stopping']['initial_training']
        self.patience = config['optimization']['early_stopping']['patience']
        
        self.log_frequency = config['model']['log_frequency']
        self.ckpt = config['model']['ckpt']
        self.best_ckpt = config['model']['best_ckpt']

        self.mode = ''

    def fit(self):
        wandb.init(project=self.project_name)

        avg_loss = 0
        num_step = 0
        prev_r = 0
        prev_ema_r = 0

        for n_epoch in range(self.num_epoch):
            train_dataloader = DataLoader(self.train_dataset,  
                                          collate_fn=lambda x: x[0],
                                          pin_memory=True)
            for inputs, labels in train_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
            
                loss = self.train_one_step(inputs, labels)

                avg_loss += loss
                num_step += 1

                if num_step % self.log_frequency == 0:
                    os.makedirs(self.ckpt, exist_ok=True)
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'num_step': num_step}, 
                                self.ckpt + '/train.pt')

                    avg_loss = avg_loss / self.log_frequency
                    wandb.log({"train/loss": avg_loss}, 
                                step=num_step)
                    avg_loss = 0

                    valid_dataloader = DataLoader(self.valid_dataset, 
                                                  collate_fn=lambda x: x[0],
                                                  pin_memory=True)
                    scores = self.do_eval(valid_dataloader)

                    r = scores['spearmanr']
                    if r > prev_r:
                        os.makedirs(self.best_ckpt, exist_ok=True)
                        torch.save({'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'num_step': num_step}, 
                                    self.best_ckpt + '/best.pt')
                        prev_r = r

                    wandb.log({"eval/spearmanr": scores['spearmanr'], 
                               "eval/mean_squared_error": scores['mean_squared_error']}, 
                                step=num_step)

            if n_epoch == 0:
                prev_ema_r = r
            
            ema_r = r * (1 - self.smoothing_factor) + prev_ema_r * self.smoothing_factor
            
            if (round(prev_ema_r, 4) >= round(ema_r, 4)) and n_epoch > self.initial_training:
                counter += 1
            else:
                counter = 0

            prev_ema_r = ema_r

            if counter > self.patience:
                break

    def train_one_step(self, inputs, labels):
        """
        Performs a single training step.

        Parameters
        ----------
        inputs (torch.Tensor): Input data.
        labels (torch.Tensor): Target labels.

        Returns
        -------
        float:
            Loss value for the step.
        """
        if self.mode != 'train':
            self.model.train()
            self.mode = 'train'

        self.optimizer.zero_grad()

        outputs = self.model(inputs)

        loss = self.criterion(outputs, labels)

        loss.backward()

        self.optimizer.step()
        self.scheduler.step()
        
        return loss
    
    def do_eval(self, dataloader):
        if self.mode != 'eval':
            self.model.eval()
            self.mode = 'eval'

        all_outputs = []
        all_labels = []

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)

            output = self.model(inputs)
            output = output.detach().cpu().numpy()
            
            all_outputs.append(output)
            all_labels.append(labels)
        
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_outputs = np.squeeze(all_outputs)

        all_labels = np.concatenate(all_labels, axis=0)
        all_labels = np.squeeze(all_labels)

        scores = {}

        scores['spearmanr'] = spearmanr(all_labels, all_outputs)[0]
        scores['mean_squared_error'] = mean_squared_error(all_labels, all_outputs)

        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    
    args = parser.parse_args()

    config = yaml_parser(args.config)

    trainer = Trainer(config)
    trainer.fit()

if __name__ == "__main__":
    main()