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
    
    return cm