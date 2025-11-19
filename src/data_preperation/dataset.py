import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class RainDataset(Dataset):
    """Dataset for rainy/clean image pairs"""
    
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_images = sorted(os.listdir(input_dir))
        self.target_images = sorted(os.listdir(target_dir))
        self.transform = transform if transform else T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        target_path = os.path.join(self.target_dir, self.target_images[idx])

        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        return self.transform(input_img), self.transform(target_img)


class PVDNCocoDataset(Dataset):
    """Dataset for PVDN in COCO format"""
    
    def __init__(self, coco_json_path, img_root, transforms=None):
        self.img_root = Path(img_root)
        self.transforms = transforms
        
        with open(coco_json_path) as f:
            self.coco = json.load(f)
        
        self.images = self.coco['images']
        self.annotations = self.coco['annotations']
        
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        print(f"Loaded {len(self.images)} images with {len(self.annotations)} annotations")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = self.img_root / img_info['file_name']
        img = Image.open(img_path).convert("RGB")
        
        img_id = img_info['id']
        anns = self.img_to_anns.get(img_id, [])
        
        boxes = []
        labels = []
        areas = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": torch.zeros((len(anns),), dtype=torch.int64)
        }
        
        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)
        
        return img, target


def collate_fn(batch):
    """Custom collate for variable boxes per image"""
    return tuple(zip(*batch))