import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

class CosineScheduler:
    def __init__(self, max_lr, total_epochs):
        self.max_lr = max_lr
        self.total_epochs = total_epochs

    def get_lr(self, epoch):
        return self.max_lr * 0.5 * (1 + np.cos(np.pi * epoch / self.total_epochs))

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Use AdamW with gradient clipping
        self.optimizer = AdamW(model.parameters(), 
                             lr=config['learning_rate'], 
                             weight_decay=config['weight_decay'])
        
        # Use OneCycleLR scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            epochs=config['num_epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Use SmoothL1Loss instead of MSE
        self.criterion = nn.SmoothL1Loss()
        
        # Initialize lists to store metrics
        self.train_losses = []
        self.val_losses = []
        self.train_r2_scores = []
        self.val_r2_scores = []

    def train(self):
        epoch_pbar = tqdm(range(self.config['num_epochs']), desc='Training Progress')
        
        for epoch in epoch_pbar:
            # Train for one epoch
            train_loss, train_r2 = self._train_epoch()
            val_loss, val_r2 = self._validate_epoch()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_r2_scores.append(train_r2)
            self.val_r2_scores.append(val_r2)
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'train_R²': f'{train_r2:.4f}',
                'val_R²': f'{val_r2:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Plot metrics after training
        self.plot_metrics()

    def _train_epoch(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        batch_pbar = tqdm(enumerate(self.train_loader), 
                         total=len(self.train_loader),
                         desc='Training',
                         leave=False)
        
        for batch_idx, (data, target) in batch_pbar:
            # Move data to device
            data = data.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Move predictions and targets to CPU for metrics calculation
            all_preds.extend(output.detach().cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            batch_pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        r2 = r2_score(all_targets, all_preds)
        
        return avg_loss, r2

    def _validate_epoch(self):
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                # Move data to device
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                # Move predictions and targets to CPU for metrics calculation
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = val_loss / len(self.val_loader)
        r2 = r2_score(all_targets, all_preds)
        
        return avg_loss, r2

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('SmoothL1 Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot R² scores
        ax2.plot(epochs, self.train_r2_scores, 'b-', label='Training R²')
        ax2.plot(epochs, self.val_r2_scores, 'r-', label='Validation R²')
        ax2.set_title('Training and Validation R² Score')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('R² Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()