import numpy as np
from scipy.stats import norm
import pandas as pd

class DataProcessor:
    def __init__(self):
        # Update feature columns based on the printed columns
        self.feature_columns = [
            'M1', 'M2', 'mu', 'dL', 'dR', 'uL', 'uR',
            'chi10', 'chi1pm', 'chi20', 'chi2pm', 'chi30', 'chi40',
            'tau1', 'g', 't1', 'b1', 'Mq1'
        ]
        
        self.target_columns = [
    'bf_cms_sus_19_006_mu1p0f',
       'bf_cms_sus_20_001_mu1p0s', 'bf_cms_sus_21_006_mu1p0f',
       'bf_cms_sus_18_004_mu1p0f', 'bf_cms_sus_21_007_mb_mu1p0s'
        ]

    def load_parquet_data(self, file_path):
        """Load data from parquet file"""
        data = pd.read_parquet(file_path)
        return data

    def split_data(self, data, train_fraction=0.8, val_fraction=0.1):
        """Split data into train/val/test sets"""
        n_samples = len(data)
        indices = np.random.permutation(n_samples)
        
        train_size = int(train_fraction * n_samples)
        val_size = int(val_fraction * n_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return {
            'train': data.iloc[train_indices],
            'val': data.iloc[val_indices],
            'test': data.iloc[test_indices]
        }

    def process_features(self, data):
        """Process features from the dataframe"""
        processed_data = {}
        
        for split, df in data.items():
            # Select features and convert to numpy array with float32 dtype
            features = df[self.feature_columns].to_numpy(dtype=np.float32)
            
            # Apply log transformation to mass parameters (all positive values)
            features = np.log10(np.abs(features) + 1)
            
            # Apply Gaussian rank transformation to each feature column
            processed_features = np.zeros_like(features, dtype=np.float32)
            for i in range(features.shape[1]):
                processed_features[:, i] = self.gaussian_rank_transform(features[:, i])
            
            processed_data[split] = processed_features
            
        return processed_data

    def process_targets(self, data):
        """Process target variables from the dataframe"""
        processed_targets = {}
        
        for split, df in data.items():
            # Convert targets to float32
            targets = df[self.target_columns].to_numpy(dtype=np.float32)
            
            # Add small epsilon to handle zeros before log transform
            epsilon = 1e-20
            targets = np.log10(targets + epsilon)
            
            # Apply Gaussian rank transformation to each target column
            processed_targets_array = np.zeros_like(targets, dtype=np.float32)
            for i in range(targets.shape[1]):
                processed_targets_array[:, i] = self.gaussian_rank_transform(targets[:, i])
            
            processed_targets[split] = processed_targets_array
            
        return processed_targets

    @staticmethod
    def gaussian_rank_transform(x):
        """Apply Gaussian rank transformation to a single feature"""
        ranks = np.argsort(np.argsort(x))
        return norm.ppf((ranks + 1)/(len(ranks) + 1))

    def preprocess_features(self, features):
        """Preprocess features with appropriate transformations"""
        # Apply log transformation to mass parameters (all positive values)
        features = np.log10(np.abs(features) + 1)
        
        # Normalize all features
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        
        return features

    @staticmethod
    def normalize(x):
        """Normalize features"""
        return (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)

    @staticmethod
    def gaussian_rank_transform(x):
        ranks = np.argsort(np.argsort(x))
        return norm.ppf((ranks + 1)/(len(ranks) + 1))

    @staticmethod
    def normalize(x):
        return (x - np.mean(x)) / np.std(x)
