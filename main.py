import torch
from model import ResidualNetwork
from data_processor import DataProcessor
from trainer import Trainer
from utils import load_config, PhysicsDataset, plot_predictions
from torch.utils.data import DataLoader
import numpy as np
import os

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration with correct path to config directory
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    config = load_config(config_path)
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Load data from parquet file
    raw_data = data_processor.load_parquet_data('bf_pred.parquet')
    print("Available columns in the parquet file:")
    print(raw_data.columns.tolist())
    
    # Split the data
    split_data = data_processor.split_data(raw_data)
    
    # Process features and targets
    processed_features = data_processor.process_features(split_data)
    processed_targets = data_processor.process_targets(split_data)
    
    # Create datasets
    train_dataset = PhysicsDataset(processed_features['train'], processed_targets['train'])
    val_dataset = PhysicsDataset(processed_features['val'], processed_targets['val'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, 
                            batch_size=config['training']['batch_size'],
                            shuffle=True,
                            pin_memory=True)  # Enable faster data transfer to GPU
    val_loader = DataLoader(val_dataset,
                          batch_size=config['training']['batch_size'],
                          shuffle=False,
                          pin_memory=True)  # Enable faster data transfer to GPU
    
    # Initialize model
    model = ResidualNetwork(
        input_dim=len(data_processor.feature_columns),
        output_dim=len(data_processor.target_columns)
    ).to(device)  # Move model to GPU
    
    # Initialize trainer with device
    trainer = Trainer(model, train_loader, val_loader, config['training'], device)
    
    # Train the model
    trainer.train()

    # Plot predictions
    X_val = torch.tensor(processed_features['val'], dtype=torch.float32, device=device)
    y_val = torch.tensor(processed_targets['val'], dtype=torch.float32, device=device)
    plot_predictions(model, X_val, y_val)

if __name__ == "__main__":
    main()