"""
Lightweight SwinIR for Rain Removal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttention(nn.Module):
    """Memory-efficient channel attention"""
    def __init__(self, dim):
        super(SimpleAttention, self).__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        chunk_size = 64
        outputs = []
        
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            x_chunk = x[:, i:end_i, :]
            
            qkv = self.qkv(x_chunk).reshape(B, end_i - i, 3, C).permute(2, 0, 1, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            
            out_chunk = (attn @ v)
            outputs.append(out_chunk)
        
        out = torch.cat(outputs, dim=1)
        out = self.proj(out)
        return out


class ConvFFN(nn.Module):
    """Convolutional Feed-Forward Network"""
    def __init__(self, dim, hidden_dim):
        super(ConvFFN, self).__init__()
        self.conv1 = nn.Conv2d(dim, hidden_dim, 1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, dim, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.conv3(x)
        return x


class SwinBlock_Light(nn.Module):
    """Lightweight Swin transformer block"""
    def __init__(self, dim, mlp_ratio=2.):
        super(SwinBlock_Light, self).__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimpleAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = ConvFFN(dim, hidden_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        shortcut = x
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.norm1(x_flat)
        x_flat = self.attn(x_flat, H, W)
        x_flat = x_flat.transpose(1, 2).reshape(B, C, H, W)
        x = shortcut + x_flat
        
        x = x + self.ffn(self.norm2(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W))
        
        return x


class ResidualGroup(nn.Module):
    """Group of transformer blocks with residual connection"""
    def __init__(self, dim, num_blocks):
        super(ResidualGroup, self).__init__()
        self.blocks = nn.ModuleList([
            SwinBlock_Light(dim) for _ in range(num_blocks)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
    
    def forward(self, x):
        shortcut = x
        for blk in self.blocks:
            x = blk(x)
        x = self.conv(x)
        return x + shortcut


class SwinIR_Light(nn.Module):
    """Lightweight SwinIR for Rain Removal"""
    
    def __init__(self, in_channels=3, out_channels=3, dim=32, num_blocks=3, num_groups=3):
        super(SwinIR_Light, self).__init__()
        
        self.conv_first = nn.Conv2d(in_channels, dim, 3, 1, 1)
        
        self.groups = nn.ModuleList([
            ResidualGroup(dim, num_blocks) for _ in range(num_groups)
        ])
        
        self.conv_after_body = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(dim, out_channels, 3, 1, 1)
    
    def forward(self, x):
        x_first = self.conv_first(x)
        
        x = x_first
        for group in self.groups:
            x = group(x)
        
        x = self.conv_after_body(x) + x_first
        x = self.conv_last(x)
        return x