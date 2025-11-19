import os
import random
import math
import argparse
from pathlib import Path
import cv2
import numpy as np


def add_perspective_rain(img, num_streaks=1500, perspective_strength=0.15,
                         max_thickness=2, intensity=220, blur_kernel_size=(3, 3)):
    """Add synthetic rain with perspective effect"""
    h, w, _ = img.shape
    rain_layer = np.zeros_like(img, dtype=np.uint8)
    vanishing_point = (w // 2, h // 2)

    for _ in range(num_streaks):
        end_x = np.random.randint(0, w)
        end_y = np.random.randint(0, h)

        distance = math.sqrt((end_x - vanishing_point[0])**2 + 
                           (end_y - vanishing_point[1])**2)

        length = distance * perspective_strength
        thickness = 1 + int((distance / max(w, h)) * (max_thickness - 1))

        angle = math.atan2(end_y - vanishing_point[1], 
                          end_x - vanishing_point[0])

        start_x = int(end_x - length * math.cos(angle))
        start_y = int(end_y - length * math.sin(angle))

        cv2.line(rain_layer, (start_x, start_y), (end_x, end_y), 
                (intensity, intensity, intensity), thickness)

    if blur_kernel_size[0] > 1:
        rain_layer = cv2.GaussianBlur(rain_layer, blur_kernel_size, 0)
    
    rainy = cv2.addWeighted(img, 1.0, rain_layer, 0.7, 0)
    return rainy


def process_scenes(scene_list, output_dir, sample_fraction=1.0, val_split=0.1,
                   rain_params=None):
    """Process multiple scene directories to generate rain dataset"""
    if rain_params is None:
        rain_params = {
            'num_streaks': 1500,
            'perspective_strength': 0.15,
            'max_thickness': 2,
            'intensity': 220,
            'blur_kernel_size': (3, 3)
        }
    
    output_dir = Path(output_dir)
    
    for split in ("train", "val"):
        (output_dir / split / "input").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "target").mkdir(parents=True, exist_ok=True)
    
    idx_train = 0
    idx_val = 0
    
    print("="*60)
    print("SYNTHETIC RAIN DATASET GENERATION")
    print("="*60)
    print(f"Rain parameters:")
    for k, v in rain_params.items():
        print(f"  {k}: {v}")
    print()
    
    for scene in scene_list:
        scene_path = Path(scene)
        files = [p for p in sorted(scene_path.glob("*")) 
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        
        if not files:
            print(f"⚠️  No images in {scene}, skipping.")
            continue
        
        print(f"Processing: {scene_path.name} ({len(files)} images)")
        
        n_sample = max(1, int(len(files) * sample_fraction))
        sampled = random.sample(files, n_sample)
        random.shuffle(sampled)
        
        n_val = int(len(sampled) * val_split)
        val_files = sampled[:n_val]
        train_files = sampled[n_val:]
        
        for fp in train_files:
            clean = cv2.imread(str(fp))
            if clean is None:
                continue
            rainy = add_perspective_rain(clean, **rain_params)
            
            cv2.imwrite(str(output_dir / "train" / "target" / f"{idx_train:06d}.png"), 
                       clean)
            cv2.imwrite(str(output_dir / "train" / "input" / f"{idx_train:06d}.png"), 
                       rainy)
            idx_train += 1
        
        for fp in val_files:
            clean = cv2.imread(str(fp))
            if clean is None:
                continue
            rainy = add_perspective_rain(clean, **rain_params)
            
            cv2.imwrite(str(output_dir / "val" / "target" / f"{idx_val:06d}.png"), 
                       clean)
            cv2.imwrite(str(output_dir / "val" / "input" / f"{idx_val:06d}.png"), 
                       rainy)
            idx_val += 1
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Train pairs: {idx_train}")
    print(f"Val pairs: {idx_val}")
    print(f"Dataset root: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic rain dataset")
    parser.add_argument("--scenes_dir", type=str, required=True,
                        help="Directory containing scene folders")
    parser.add_argument("--output_dir", type=str, default="./synthetic_rain_dataset",
                        help="Output directory")
    parser.add_argument("--sample_fraction", type=float, default=1.0,
                        help="Fraction of images to use per scene")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction for validation set")
    parser.add_argument("--num_streaks", type=int, default=1500,
                        help="Number of rain streaks")
    parser.add_argument("--perspective_strength", type=float, default=0.15,
                        help="Perspective effect strength")
    parser.add_argument("--max_thickness", type=int, default=2,
                        help="Maximum streak thickness")
    parser.add_argument("--intensity", type=int, default=220,
                        help="Rain brightness (0-255)")
    parser.add_argument("--blur_size", type=int, default=3,
                        help="Motion blur kernel size")
    
    args = parser.parse_args()
    
    if ',' in args.scenes_dir:
        scene_list = [s.strip() for s in args.scenes_dir.split(',')]
    else:
        scenes_root = Path(args.scenes_dir)
        if scenes_root.is_dir():
            scene_list = [str(p) for p in scenes_root.iterdir() if p.is_dir()]
        else:
            scene_list = [args.scenes_dir]
    
    rain_params = {
        'num_streaks': args.num_streaks,
        'perspective_strength': args.perspective_strength,
        'max_thickness': args.max_thickness,
        'intensity': args.intensity,
        'blur_kernel_size': (args.blur_size, args.blur_size)
    }
    
    process_scenes(scene_list, args.output_dir, args.sample_fraction,
                  args.val_split, rain_params)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()