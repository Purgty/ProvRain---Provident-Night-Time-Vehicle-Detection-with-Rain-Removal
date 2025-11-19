import torch
import torch.nn as nn
import torch.nn.functional as F


class MDTA_Light(nn.Module):
    """Lightweight Multi-Dconv Head Transposed Attention"""
    def __init__(self, channels, num_heads=2):
        super(MDTA_Light, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, 
                                     padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        out = out.reshape(b, -1, h, w)
        out = self.project_out(out)
        return out


class GDFN_Light(nn.Module):
    """Lightweight Gated Feed-Forward Network"""
    def __init__(self, channels, expansion_factor=2.0):
        super(GDFN_Light, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, 
                               padding=1, groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock_Light(nn.Module):
    """Lightweight Transformer Block"""
    def __init__(self, channels, num_heads=2, expansion_factor=2.0):
        super(TransformerBlock_Light, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA_Light(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN_Light(channels, expansion_factor)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        x_norm = self.norm1(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b, c, h, w)
        x = x + self.attn(x_norm)
        
        x_norm = self.norm2(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b, c, h, w)
        x = x + self.ffn(x_norm)
        
        return x


class Restormer_Light(nn.Module):
    """Lightweight Restormer for Rain Removal"""
    
    def __init__(self, in_channels=3, out_channels=3, channels=24,
                 num_blocks=[2, 2, 2, 3], num_heads=[1, 2, 2, 4]):
        super(Restormer_Light, self).__init__()
        
        self.embed = nn.Conv2d(in_channels, channels, 3, padding=1)
        
        # Encoder
        self.encoder1 = nn.Sequential(*[TransformerBlock_Light(channels, num_heads[0]) 
                                        for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1)
        
        self.encoder2 = nn.Sequential(*[TransformerBlock_Light(channels * 2, num_heads[1]) 
                                        for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(channels * 2, channels * 4, 4, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(*[TransformerBlock_Light(channels * 4, num_heads[2]) 
                                          for _ in range(num_blocks[2])])
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, 2, stride=2)
        self.reduce2 = nn.Conv2d(channels * 4, channels * 2, 1)
        self.decoder2 = nn.Sequential(*[TransformerBlock_Light(channels * 2, num_heads[1]) 
                                        for _ in range(num_blocks[1])])
        
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.reduce1 = nn.Conv2d(channels * 2, channels, 1)
        self.decoder1 = nn.Sequential(*[TransformerBlock_Light(channels, num_heads[0]) 
                                        for _ in range(num_blocks[0])])
        
        self.output = nn.Conv2d(channels, out_channels, 3, padding=1)
    
    def forward(self, x):
        x = self.embed(x)
        
        # Encoder
        e1 = self.encoder1(x)
        x = self.down1(e1)
        
        e2 = self.encoder2(x)
        x = self.down2(e2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up2(x)
        x = self.decoder2(self.reduce2(torch.cat([x, e2], dim=1)))
        
        x = self.up1(x)
        x = self.decoder1(self.reduce1(torch.cat([x, e1], dim=1)))
        
        return self.output(x)