import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data_preparation.dataset import PVDNCocoDataset, collate_fn
from ..models.detection.faster_rcnn import get_faster_rcnn_model
from ..utils.visualization import plot_training_history


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        pbar.set_postfix({'loss': f'{losses.item():.4f}'})
    
    return total_loss / len(data_loader)


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate on validation set"""
    model.train()
    total_loss = 0
    
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
    
    return total_loss / len(data_loader)


def train_faster_rcnn(args):
    """Train Faster R-CNN model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create datasets
    train_dataset = PVDNCocoDataset(args.train_json, args.train_img_root)
    val_dataset = PVDNCocoDataset(args.val_json, args.val_img_root)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers,
                             collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers,
                           collate_fn=collate_fn)
    
    # Create model
    model = get_faster_rcnn_model(num_classes=3, pretrained=True)
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, 
                         weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, 
                                         gamma=args.scheduler_gamma)
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate(model, val_loader, device)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.models_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.models_dir, 'faster_rcnn_best.pth'))
            print("  ✓ Best model saved!")
        
        print("-"*60)
    
    # Plot history
    os.makedirs(args.results_dir, exist_ok=True)
    plot_training_history(history, "Faster_RCNN", 
                         os.path.join(args.results_dir, 'faster_rcnn_training.png'))
    
    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN")
    
    # Data
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--train_img_root", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--val_img_root", type=str, required=True)
    
    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--scheduler_step", type=int, default=5)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    
    # Paths
    parser.add_argument("--models_dir", type=str, default="models/saved_models")
    parser.add_argument("--results_dir", type=str, default="results/figures")
    
    args = parser.parse_args()
    train_faster_rcnn(args)


if __name__ == "__main__":
    main()