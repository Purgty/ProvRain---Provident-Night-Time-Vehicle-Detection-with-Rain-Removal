import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def visualize_predictions(model, val_loader, device, num_samples=5, score_threshold=0.5):
    """Visualize model predictions on validation set"""
    model.eval()
    
    category_names = {1: 'Vehicle', 2: 'Light'}
    colors = {1: 'red', 2: 'yellow'}
    
    images_list = []
    targets_list = []
    
    for images, targets in val_loader:
        images_list.extend(images)
        targets_list.extend(targets)
        if len(images_list) >= num_samples:
            break
    
    images_list = images_list[:num_samples]
    targets_list = targets_list[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(16, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        img = images_list[idx].to(device)
        target = targets_list[idx]
        
        # Prediction
        with torch.no_grad():
            pred = model([img])[0]
        
        img_np = img.permute(1, 2, 0).cpu().numpy()
        
        # Ground Truth
        ax = axes[idx, 0]
        ax.imshow(img_np)
        ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=colors[label.item()],
                facecolor='none'
            )
            ax.add_patch(rect)
        
        # Predictions
        ax = axes[idx, 1]
        ax.imshow(img_np)
        ax.set_title(f'Predictions (threshold={score_threshold})', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        keep = pred['scores'] > score_threshold
        boxes = pred['boxes'][keep].cpu()
        labels = pred['labels'][keep].cpu()
        scores = pred['scores'][keep].cpu()
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=colors[label.item()],
                facecolor='none'
            )
            ax.add_patch(rect)
            
            ax.text(x1, y1-5, f'{category_names[label.item()]}: {score:.2f}',
                   color='white', fontsize=9, fontweight='bold',
                   bbox=dict(facecolor=colors[label.item()], alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_training_history(history, model_name, save_path=None):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'], label='Val', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    if 'psnr' in history:
        axes[1].plot(history['psnr'], marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('PSNR (dB)')
        axes[1].set_title(f'{model_name} - PSNR')
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix"""
    class_names = ['Background', 'Vehicle', 'Light']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig