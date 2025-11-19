import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    DnCNN for image denoising/deraining.
    Architecture: Conv + (Conv-BN-ReLU) x depth + Conv
    Learns residual (noise/rain) and subtracts from input.
    """
    def __init__(self, channels=3, num_layers=17, features=64):
        super(DnCNN, self).__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Conv2d(channels, features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        # Last layer
        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        residual = self.dncnn(x)
        return x - residual