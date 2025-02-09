import numpy as np
from scipy.stats import norm

class DataProcessor:
    @staticmethod
    def gaussian_rank_transform(x):
        ranks = np.argsort(np.argsort(x))
        return norm.ppf((ranks + 1)/(len(ranks) + 1))

    @staticmethod
    def normalize(x):
        return (x - np.mean(x)) / np.std(x)

    def process_features(self, data):
        processed_data = {}
        
        #adjsut for desired observables
        processed_data['mass'] = self.gaussian_rank_transform(data['mass'])
        processed_data['mass_differences'] = np.log10(data['mass_differences'])
        processed_data['mixing_angle'] = data['mixing_angle']
        processed_data['branching_ratio'] = data['branching_ratio']
        processed_data['decay_length'] = np.log10(data['decay_length'])
        processed_data['single_region_border'] = self.normalize(data['single_region_border'])
        
        return processed_data