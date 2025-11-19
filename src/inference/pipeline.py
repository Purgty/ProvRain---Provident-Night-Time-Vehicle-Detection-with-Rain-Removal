import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from ..models.denoising.enhanced_unet import EnhancedUNet
from ..models.detection.faster_rcnn import get_faster_rcnn_model


def run_pipeline(args):
    """Run full pipeline"""
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load denoiser
    print("Loading denoiser...")
    denoiser = EnhancedUNet(return_mask=False).to(device)
    checkpoint = torch.load(args.denoiser_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        denoiser.load_state_dict(checkpoint['model_state_dict'])
    else:
        denoiser.load_state_dict(checkpoint)
    denoiser.eval()
    print("✓ Denoiser loaded\n")
    
    # Load detector
    print("Loading detector...")
    detector = get_faster_rcnn_model(num_classes=3, pretrained=False)
    checkpoint = torch.load(args.detector_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        detector.load_state_dict(checkpoint['model_state_dict'])
    else:
        detector.load_state_dict(checkpoint)
    detector.to(device)
    detector.eval()
    print("✓ Detector loaded\n")
    
    # Load image
    transform_denoise = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    transform_detect = T.ToTensor()
    
    input_img = Image.open(args.image_path).convert("RGB")
    
    # Step 1: Denoise
    print("Step 1: Denoising...")
    input_tensor = transform_denoise(input_img).unsqueeze(0).to(device)
    with torch.no_grad():
        denoised_tensor = denoiser(input_tensor)
    denoised_img = denoised_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
    denoised_pil = Image.fromarray((denoised_img * 255).astype('uint8'))
    print("✓ Denoising complete\n")
    
    # Step 2: Detect
    print("Step 2: Detecting vehicles...")
    detect_tensor = transform_detect(denoised_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = detector(detect_tensor)[0]
    
    keep = predictions['scores'] > args.score_threshold
    boxes = predictions['boxes'][keep].cpu()
    labels = predictions['labels'][keep].cpu()
    scores = predictions['scores'][keep].cpu()
    
    print(f"✓ Detected {len(boxes)} objects")
    print(f"  Vehicles: {(labels == 1).sum().item()}")
    print(f"  Lights: {(labels == 2).sum().item()}\n")
    
    # Visualize
    os.makedirs(args.output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original
    axes[0].imshow(input_img)
    axes[0].set_title('Input (Rainy)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Denoised
    axes[1].imshow(denoised_img)
    axes[1].set_title('Denoised', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Detected
    axes[2].imshow(denoised_img)
    axes[2].set_title(f'Detection (threshold={args.score_threshold})', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    category_names = {1: 'Vehicle', 2: 'Light'}
    colors = {1: 'red', 2: 'yellow'}
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=colors[label.item()],
            facecolor='none'
        )
        axes[2].add_patch(rect)
        
        axes[2].text(x1, y1-5, f'{category_names[label.item()]}: {score:.2f}',
                    color='white', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor=colors[label.item()], alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, 'pipeline_result.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline")
    parser.add_argument("--denoiser_path", type=str, required=True)
    parser.add_argument("--detector_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/pipeline")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()