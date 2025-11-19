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