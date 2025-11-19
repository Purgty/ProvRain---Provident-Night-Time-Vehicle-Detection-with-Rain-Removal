import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

from ..models.denoising.enhanced_unet import EnhancedUNet
from ..models.denoising.dncnn import DnCNN
from ..models.denoising.unet import UNet
from ..models.denoising.resdenoiser import ResDenoiser
from ..models.denoising.swinir import SwinIR_Light
from ..models.denoising.restormer import Restormer_Light


def sharpen_image(output_img):
    """Apply sharpening filter"""
    img_uint8 = (output_img.clip(0, 1) * 255).astype(np.uint8)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img_uint8, -1, kernel)
    return sharpened.astype(np.float32) / 255.0


def denoise_image(args):
    """Denoise a single image"""
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print(f"Loading {args.model_type} from {args.model_path}...")
    
    model_dict = {
        'dncnn': DnCNN,
        'unet': UNet,
        'enhanced_unet': EnhancedUNet,
        'resdenoiser': ResDenoiser,
        'swinir': SwinIR_Light,
        'restormer': Restormer_Light
    }
    
    model_class = model_dict[args.model_type]
    
    if args.model_type == 'enhanced_unet':
        model = model_class(return_mask=False).to(device)
    else:
        model = model_class().to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✓ Model loaded\n")
    
    # Load image
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    input_img = Image.open(args.image_path).convert("RGB")
    input_tensor = transform(input_img).unsqueeze(0).to(device)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        output = model(input_tensor)
    
    output_img = output.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
    
    # Sharpen if requested
    if args.sharpen:
        output_img = sharpen_image(output_img)
    
    # Save
    output_pil = Image.fromarray((output_img * 255).astype(np.uint8))
    output_pil.save(args.output_path)
    
    print(f"✓ Saved to: {args.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Denoise image")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="enhanced_unet",
                       choices=['dncnn', 'unet', 'enhanced_unet', 'resdenoiser', 
                               'swinir', 'restormer'])
    parser.add_argument("--sharpen", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    denoise_image(args)


if __name__ == "__main__":
    main()