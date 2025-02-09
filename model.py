import torch
import torch.nn as nn

class ResidualNetwork(nn.Module):
    def __init__(self, input_dim=31):
        super(ResidualNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(3):
            layer = nn.Linear(input_dim, input_dim)
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            self.layers.append(layer)
            self.layers.append(nn.BatchNorm1d(input_dim))
            self.layers.append(nn.ReLU())

    def forward(self, x):
        identity = x
        for layer in self.layers:
            x = layer(x)
        return x + identity