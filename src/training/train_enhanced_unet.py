import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data_preparation.dataset import RainDataset
from ..models.denoising.enhanced_unet import EnhancedUNet
from ..utils.losses import RainRemovalLoss
from ..utils.visualization import plot_training_history


def train_enhanced_unet(args):
    """Train Enhanced U-Net model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create datasets
    train_dataset = RainDataset(args.train_input, args.train_target)
    val_dataset = RainDataset(args.val_input, args.val_target)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images\n")
    
    # Create model
    model = EnhancedUNet().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = RainRemovalLoss()
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for noisy, clean in pbar:
            noisy, clean = noisy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            output, rain_mask = model(noisy)
            
            loss, loss_dict = criterion(output, clean, rain_mask, noisy)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                output, rain_mask = model(noisy)
                loss, _ = criterion(output, clean, rain_mask, noisy)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch}/{args.epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(args.models_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, os.path.join(args.models_dir, 'enhanced_unet_best.pth'))
            print(f"  ✓ Best model saved!")
    
    # Plot and save history
    os.makedirs(args.results_dir, exist_ok=True)
    plot_training_history(history, "EnhancedUNet", 
                         os.path.join(args.results_dir, 'enhanced_unet_training.png'))
    
    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced U-Net")
    
    # Data
    parser.add_argument("--train_input", type=str, required=True)
    parser.add_argument("--train_target", type=str, required=True)
    parser.add_argument("--val_input", type=str, required=True)
    parser.add_argument("--val_target", type=str, required=True)
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    
    # Paths
    parser.add_argument("--models_dir", type=str, default="models/saved_models")
    parser.add_argument("--results_dir", type=str, default="results/figures")
    
    args = parser.parse_args()
    train_enhanced_unet(args)


if __name__ == "__main__":
    main()