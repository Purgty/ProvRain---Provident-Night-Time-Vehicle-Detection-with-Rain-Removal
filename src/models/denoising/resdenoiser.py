import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)


class ResDenoiser(nn.Module):
    """Residual denoising network"""
    
    def __init__(self, channels=3, num_blocks=8, features=64):
        super(ResDenoiser, self).__init__()
        
        self.conv_in = nn.Sequential(
            nn.Conv2d(channels, features, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.res_blocks = nn.Sequential(
            *[ResBlock(features) for _ in range(num_blocks)]
        )
        
        self.conv_out = nn.Conv2d(features, channels, 3, padding=1)
    
    def forward(self, x):
        feat = self.conv_in(x)
        feat = self.res_blocks(feat)
        residual = self.conv_out(feat)
        return x - residual