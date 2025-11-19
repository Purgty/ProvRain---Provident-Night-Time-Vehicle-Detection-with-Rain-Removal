import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..models.detection.faster_rcnn import get_faster_rcnn_model


def detect_vehicles(args):
    """Detect vehicles in image"""
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = get_faster_rcnn_model(num_classes=3, pretrained=False)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("✓ Model loaded\n")
    
    # Load image
    transform = T.ToTensor()
    input_img = Image.open(args.image_path).convert("RGB")
    input_tensor = transform(input_img).unsqueeze(0).to(device)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        predictions = model(input_tensor)[0]
    
    # Filter predictions
    keep = predictions['scores'] > args.score_threshold
    boxes = predictions['boxes'][keep].cpu()
    labels = predictions['labels'][keep].cpu()
    scores = predictions['scores'][keep].cpu()
    
    print(f"✓ Detected {len(boxes)} objects\n")
    print(f"Vehicles: {(labels == 1).sum().item()}")
    print(f"Lights: {(labels == 2).sum().item()}\n")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(input_img)
    ax.axis('off')
    
    category_names = {1: 'Vehicle', 2: 'Light'}
    colors = {1: 'red', 2: 'yellow'}
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=colors[label.item()],
            facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.text(x1, y1-5, f'{category_names[label.item()]}: {score:.2f}',
               color='white', fontsize=10, fontweight='bold',
               bbox=dict(facecolor=colors[label.item()], alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {args.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect vehicles")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    detect_vehicles(args)


if __name__ == "__main__":
    main()