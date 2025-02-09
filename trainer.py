import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np

class CosineScheduler:
    def __init__(self, max_lr, total_epochs):
        self.max_lr = max_lr
        self.total_epochs = total_epochs

    def get_lr(self, epoch):
        return self.max_lr * 0.5 * (1 + np.cos(np.pi * epoch / self.total_epochs))

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.optimizer = AdamW(model.parameters(), 
                             lr=config['learning_rate'], 
                             weight_decay=config['weight_decay'])
        self.scheduler = CosineScheduler(config['learning_rate'], 
                                       config['num_epochs'])
        self.criterion = nn.MSELoss()

    def train(self):
        for epoch in range(self.config['num_epochs']):
            self._train_epoch(epoch)
            val_loss = self._validate_epoch()
            
            # Add early stopping logic here if needed
            
    def _train_epoch(self, epoch):
        self.model.train()
        current_lr = self.scheduler.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
            
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

    def _validate_epoch(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
        return val_loss / len(self.val_loader)