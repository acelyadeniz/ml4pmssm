import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

class PhysicsDataset(Dataset):
    def __init__(self, data, targets):
        # Convert data to float32
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_predictions(model, X_val_tensor, y_val_tensor):
    """
    Plots the model's predictions against the actual values.

    Args:
        model (torch.nn.Module): Trained neural network model.
        X_val_tensor (torch.Tensor): Input validation data.
        y_val_tensor (torch.Tensor): Target validation data.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(X_val_tensor).cpu().numpy()
        y_actual = y_val_tensor.cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, predictions, alpha=0.5, label="Predicted vs. Actual", color='blue')
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2, label="Ideal Prediction")
    plt.xlabel("Actual Likelihood")
    plt.ylabel("Predicted Likelihood")
    plt.title("Neural Network Predictions vs Actual Likelihood")
    plt.legend()
    plt.grid(True)
    plt.show()