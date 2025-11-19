"""
Enhanced U-Net for Night-Time Rain Removal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_ch, out_ch):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)
        
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return out


class MultiScaleBlock(nn.Module):
    """Multi-Scale Feature Extraction using Dilated Convolutions"""
    def __init__(self, in_ch, out_ch):
        super(MultiScaleBlock, self).__init__()
        mid_ch = out_ch // 4
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=2, dilation=2),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=4, dilation=4),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=8, dilation=8),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.conv_fuse(out)
        return out


class EnhancedUNet(nn.Module):
    """Enhanced U-Net for night-time rain removal"""
    
    def __init__(self, in_channels=3, out_channels=3, features=64, return_mask=True):
        super(EnhancedUNet, self).__init__()
        self.return_mask = return_mask
        
        # Encoder
        self.enc1 = ResidualBlock(in_channels, features)
        self.att1 = ChannelAttention(features)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ResidualBlock(features, features * 2)
        self.att2 = ChannelAttention(features * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ResidualBlock(features * 2, features * 4)
        self.att3 = ChannelAttention(features * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ResidualBlock(features * 4, features * 8)
        self.att4 = ChannelAttention(features * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = MultiScaleBlock(features * 8, features * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 2, stride=2)
        self.dec4 = ResidualBlock(features * 16, features * 8)
        self.spatt4 = SpatialAttention()
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, stride=2)
        self.dec3 = ResidualBlock(features * 8, features * 4)
        self.spatt3 = SpatialAttention()
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, stride=2)
        self.dec2 = ResidualBlock(features * 4, features * 2)
        self.spatt2 = SpatialAttention()
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, stride=2)
        self.dec1 = ResidualBlock(features * 2, features)
        self.spatt1 = SpatialAttention()
        
        # Output
        self.out = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, out_channels, 1)
        )
        
        self.rain_mask = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e1 = self.att1(e1)
        
        e2 = self.enc2(self.pool1(e1))
        e2 = self.att2(e2)
        
        e3 = self.enc3(self.pool2(e2))
        e3 = self.att3(e3)
        
        e4 = self.enc4(self.pool3(e3))
        e4 = self.att4(e4)
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d4 = self.spatt4(d4) * d4
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d3 = self.spatt3(d3) * d3
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d2 = self.spatt2(d2) * d2
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        d1 = self.spatt1(d1) * d1
        
        # Output
        clean = self.out(d1)
        mask = self.rain_mask(d1)
        
        if self.return_mask:
            return clean


---

## File: `src/models/denoising/resdenoiser.py`
```python
"""
ResNet-style denoiser
"""

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
```

---

## File: `src/models/denoising/swinir.py`
```python
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
```

---

## File: `src/models/denoising/restormer.py`
```python
"""
Lightweight Restormer for Rain Removal
"""

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
```

---

## File: `src/models/detection/__init__.py`
```python
"""
Detection model architectures
"""

from .faster_rcnn import get_faster_rcnn_model

__all__ = ['get_faster_rcnn_model']
```

---

## File: `src/models/detection/faster_rcnn.py`
```python
"""
Faster R-CNN for vehicle detection
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_faster_rcnn_model(num_classes=3, pretrained=True):
    """
    Create Faster R-CNN with ResNet50-FPN
    
    Args:
        num_classes: 3 (background + vehicle + light_instance)
        pretrained: Use COCO pretrained weights
    
    Returns:
        model: Faster R-CNN model
    """
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    
    # Replace classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
```

---

## File: `src/utils/__init__.py`
```python
"""
Utility functions
"""

from .metrics import evaluate_metrics, calculate_map, create_confusion_matrix
from .losses import RainRemovalLoss
from .visualization import visualize_predictions, plot_training_history

__all__ = [
    'evaluate_metrics',
    'calculate_map',
    'create_confusion_matrix',
    'RainRemovalLoss',
    'visualize_predictions',
    'plot_training_history'
]
```

---

## File: `src/utils/losses.py`
```python
"""
Custom loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RainRemovalLoss(nn.Module):
    """Combined loss function for rain removal"""
    
    def __init__(self, lambda_mse=1.0, lambda_perceptual=0.5, lambda_mask=0.3):
        super(RainRemovalLoss, self).__init__()
        self.lambda_mse = lambda_mse
        self.lambda_perceptual = lambda_perceptual
        self.lambda_mask = lambda_mask
    
    def forward(self, pred, target, rain_mask=None, rainy_input=None):
        # MSE Loss
        mse_loss = F.mse_loss(pred, target)
        
        # Perceptual Loss (SSIM-based)
        ssim_loss = 1 - self._ssim(pred, target)
        
        # Rain Mask Loss
        mask_loss = 0
        if rain_mask is not None and rainy_input is not None:
            actual_rain = torch.abs(rainy_input - target)
            actual_rain_mask = (actual_rain.mean(dim=1, keepdim=True) > 0.1).float()
            mask_loss = F.binary_cross_entropy(rain_mask, actual_rain_mask)
        
        total_loss = (self.lambda_mse * mse_loss + 
                     self.lambda_perceptual * ssim_loss +
                     self.lambda_mask * mask_loss)
        
        return total_loss, {
            'mse': mse_loss.item(),
            'ssim': ssim_loss.item(),
            'mask': mask_loss.item() if isinstance(mask_loss, torch.Tensor) else 0
        }
    
    def _ssim(self, img1, img2, window_size=11):
        """Simplified SSIM calculation"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
```

---

## File: `src/utils/metrics.py`
```python
"""
Evaluation metrics
"""

import numpy as np
import torch
from torchvision.ops import box_iou
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def evaluate_metrics(pred_img, gt_img):
    """
    Compute PSNR, SSIM, and MSE
    
    Args:
        pred_img: Predicted image (numpy array, 0-1 range)
        gt_img: Ground truth image (numpy array, 0-1 range)
    
    Returns:
        psnr_val, ssim_val, mse_val
    """
    psnr_val = psnr(gt_img, pred_img, data_range=1.0)
    ssim_val = ssim(gt_img, pred_img, channel_axis=2, data_range=1.0)
    mse_val = np.mean((gt_img - pred_img) ** 2)
    return psnr_val, ssim_val, mse_val


def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP)
    
    Args:
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts
        iou_threshold: IoU threshold
    
    Returns:
        ap_per_class: Dict mapping class_id to AP
    """
    ap_per_class = {}
    
    for class_id in [1, 2]:  # Vehicle, Light
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []
        
        for pred, gt in zip(predictions, ground_truths):
            class_mask = pred['labels'] == class_id
            if class_mask.sum() > 0:
                all_pred_boxes.append(pred['boxes'][class_mask])
                all_pred_scores.append(pred['scores'][class_mask])
            
            gt_class_mask = gt['labels'] == class_id
            if gt_class_mask.sum() > 0:
                all_gt_boxes.append(gt['boxes'][gt_class_mask])
        
        if len(all_pred_boxes) == 0 or len(all_gt_boxes) == 0:
            ap_per_class[class_id] = 0.0
            continue
        
        pred_boxes = torch.cat(all_pred_boxes, dim=0)
        pred_scores = torch.cat(all_pred_scores, dim=0)
        
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
        
        total_gt = sum(len(gt_boxes) for gt_boxes in all_gt_boxes)
        tp = torch.zeros(len(pred_boxes))
        fp = torch.zeros(len(pred_boxes))
        
        matched_gts = [set() for _ in range(len(all_gt_boxes))]
        
        for pred_idx in range(len(pred_boxes)):
            pred_box = pred_boxes[pred_idx].unsqueeze(0)
            
            best_iou = 0
            best_gt_idx = -1
            best_img_idx = -1
            
            for img_idx, gt_boxes in enumerate(all_gt_boxes):
                if len(gt_boxes) == 0:
                    continue
                
                ious = box_iou(pred_box, gt_boxes)[0]
                max_iou, max_gt = ious.max(dim=0)
                
                if max_iou > best_iou:
                    best_iou = max_iou
                    best_gt_idx = max_gt.item()
                    best_img_idx = img_idx
            
            if best_iou >= iou_threshold and best_gt_idx not in matched_gts[best_img_idx]:
                tp[pred_idx] = 1
                matched_gts[best_img_idx].add(best_gt_idx)
            else:
                fp[pred_idx] = 1
        
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # 11-point interpolation
        ap = 0
        for t in torch.linspace(0, 1, 11):
            if torch.sum(recalls >= t) == 0:
                p = 0
            else:
                p = torch.max(precisions[recalls >= t])
            ap += p / 11
        
        ap_per_class[class_id] = ap.item()
    
    return ap_per_class


def create_confusion_matrix(model, data_loader, device, score_threshold=0.5, iou_threshold=0.5):
    """
    Create confusion matrix for detections
    
    Args:
        model: Trained model
        data_loader: Validation DataLoader
        device: cuda or cpu
        score_threshold: Confidence threshold
        iou_threshold: IoU threshold for matching
    
    Returns:
        confusion matrix as numpy array
    """
    model.eval()
    
    cm = np.zeros((3, 3), dtype=int)
    
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        predictions = model(images)
        
        for pred, target in zip(predictions, targets):
            keep = pred['scores'] > score_threshold
            pred_boxes = pred['boxes'][keep]
            pred_labels = pred['labels'][keep]
            
            gt_boxes = target['boxes'].to(device)
            gt_labels = target['labels'].to(device)
            
            if len(pred_boxes) == 0:
                for gt_label in gt_labels:
                    cm[gt_label.item()][0] += 1
                continue
            
            if len(gt_boxes) == 0:
                for pred_label in pred_labels:
                    cm[0][pred_label.item()] += 1
                continue
            
            ious = box_iou(pred_boxes, gt_boxes)
            
            matched_gt = set()
            
            for pred_idx in range(len(pred_boxes)):
                max_iou, max_gt_idx = ious[pred_idx].max(dim=0)
                
                if max_iou >= iou_threshold and max_gt_idx.item() not in matched_gt:
                    true_class = gt_labels[max_gt_idx].item()
                    pred_class = pred_labels[pred_idx].item()
                    cm[true_class][pred_class] += 1
                    matched_gt.add(max_gt_idx.item())
                else:
                    cm[0][pred_labels[pred_idx].item()] += 1
            
            for gt_idx in range(len(gt_boxes)):
                if gt_idx not in matched_gt:
                    cm[gt_labels[gt_idx].item()][0] += 1
    
    return cm, mask
        else:
            return clean