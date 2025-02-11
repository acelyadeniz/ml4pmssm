import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        
        #Scaling factor for the residual
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        residual = x
        x = self.block(x)
        return torch.relu(residual + self.scale * x)

class ResidualNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        

        dims = [input_dim, 128, 256, 512, 256, 128]
        
        # Input embedding
        self.input_block = nn.Sequential(
            nn.Linear(input_dim, dims[1]),
            nn.BatchNorm1d(dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Main network with skip connections
        self.layers = nn.ModuleList()
        
        # Encoder path
        for i in range(1, 3):
            # Dimension increasing residual block
            self.layers.append(nn.Sequential(
                ResidualBlock(dims[i]),
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
        
        # Middle blocks with constant dimension
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(dims[3]) for _ in range(3)
        ])
        
        # Decoder path with skip connections
        self.decoder = nn.ModuleList()
        for i in range(3, 1, -1):
            # Dimension decreasing residual block
            self.decoder.append(nn.Sequential(
                ResidualBlock(dims[i]),
                nn.Linear(dims[i], dims[i-1]),
                nn.BatchNorm1d(dims[i-1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
        
        self.output_block = nn.Sequential(
            nn.Linear(dims[1], dims[1] // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dims[1] // 2, output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):

        x = self.input_block(x)
        

        encoder_features = []
        

        for layer in self.layers:
            encoder_features.append(x)
            x = layer(x)
        

        for block in self.middle_blocks:
            x = block(x)
        
        # Decoder path with skip connections
        for i, decoder_layer in enumerate(self.decoder):
            # Add skip connection from encoder
            skip_feature = encoder_features[-(i+1)]
            x = decoder_layer(x)
            x = x + skip_feature  # Skip connection
            x = torch.relu(x)
        

        x = self.output_block(x)
        return x