import torch
import torch.nn as nn


class UNet(nn.Module):
    """U-Net for image deraining with skip connections"""
    
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 2, stride=2)
        self.dec4 = self._block(features * 16, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, stride=2)
        self.dec3 = self._block(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, stride=2)
        self.dec2 = self._block(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, stride=2)
        self.dec1 = self._block(features * 2, features)
        
        # Output
        self.out = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)