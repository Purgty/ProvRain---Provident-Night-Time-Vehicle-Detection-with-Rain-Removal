import os
import json
import random
import argparse
from pathlib import Path
from PIL import Image


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def bbox_from_kp(x, y, box_px, w, h):
    """Generate bounding box from keypoint"""
    x1 = max(0, int(x - box_px))
    y1 = max(0, int(y - box_px))
    x2 = min(w - 1, int(x + box_px))
    y2 = min(h - 1, int(y + box_px))
    return [x1, y1, x2 - x1, y2 - y1]


def build_id_to_path(seq_file):
    """Build mapping from image ID to relative path"""
    seqs = json.load(open(seq_file))["sequences"]
    id_to_path = {}
    for seq in seqs:
        seq_dir = seq["dir"]
        for fid in seq["image_ids"]:
            fname = f"{fid:06d}.png"
            id_to_path[fid] = f"{seq_dir}/{fname}"
    return id_to_path


def get_ids(split_file, split_name, kps_dir, sample_ratio=0.1):
    """Get image IDs for a split with optional sampling"""
    with open(split_file) as f:
        split_data = json.load(f)
    
    if split_name.endswith("train"):
        ids = split_data.get("train", [])
    elif split_name.endswith("val"):
        ids = split_data.get("val", [])
    elif split_name.endswith("test"):
        ids = split_data.get("test", [])
    else:
        ids = []
    
    if not ids:
        ids = [int(f.stem) for f in Path(kps_dir).glob("*.json")]
        print(f"⚠️  Fallback: using {len(ids)} ids from keypoint files for {split_name}")
    
    random.shuffle(ids)
    subset_size = max(1, int(len(ids) * sample_ratio))
    return ids[:subset_size]


def process_split(split_dir, split_name, out_root, box_px=32, sample_ratio=0.1):
    """Process a single split (train/val/test)"""
    print(f"\nProcessing: {split_name}")
    split_file = Path(split_dir) / "splits/default.json"
    kps_dir = Path(split_dir) / "labels/keypoints"
    seq_file = Path(split_dir) / "labels/sequences.json"
    img_root = Path(split_dir) / "images"

    ids_split = get_ids(split_file, split_name, kps_dir, sample_ratio)
    id_to_path = build_id_to_path(seq_file)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "vehicle"},
            {"id": 2, "name": "light_instance"}
        ]
    }

    ann_id, img_id = 1, 1
    ensure_dir(Path(out_root) / "coco")

    for iid in ids_split:
        kp_file = kps_dir / f"{iid:06d}.json"
        if not kp_file.exists() or iid not in id_to_path:
            continue

        img_rel = id_to_path[iid]
        img_path = img_root / img_rel
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        coco["images"].append({
            "id": img_id,
            "file_name": img_rel,
            "width": w,
            "height": h
        })

        kp_data = json.load(open(kp_file))
        
        for vehicle in kp_data.get("annotations", []):
            if "pos" not in vehicle:
                continue
            x, y = vehicle["pos"]
            bb = bbox_from_kp(x, y, box_px, w, h)
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bb,
                "area": bb[2] * bb[3],
                "iscrowd": 0,
                "vehicle_oid": vehicle.get("oid", None),
                "vehicle_direct": vehicle.get("direct", None),
                "num_instances": len(vehicle.get("instances", []))
            })
            ann_id += 1

            for inst in vehicle.get("instances", []):
                xi, yi = inst["pos"]
                bb_i = bbox_from_kp(xi, yi, box_px // 2, w, h)
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 2,
                    "bbox": bb_i,
                    "area": bb_i[2] * bb_i[3],
                    "iscrowd": 0,
                    "vehicle_oid": vehicle.get("oid", None),
                    "instance_iid": inst.get("iid", None),
                    "direct": inst.get("direct", None),
                    "rear": inst.get("rear", None)
                })
                ann_id += 1

        img_id += 1

    coco_file = Path(out_root) / "coco" / f"{split_name}.json"
    with open(coco_file, "w") as f:
        json.dump(coco, f)
    
    print(f"✅ Wrote {coco_file}")
    print(f"   Images: {len(coco['images'])} | Annotations: {len(coco['annotations'])}")
    
    return coco_file


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO dataset from PVDN")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Root directory of PVDN dataset")
    parser.add_argument("--output_dir", type=str, default="./processed",
                        help="Output directory for COCO annotations")
    parser.add_argument("--box_px", type=int, default=32,
                        help="Bounding box size in pixels")
    parser.add_argument("--sample_ratio", type=float, default=0.1,
                        help="Fraction of dataset to use (0.0-1.0)")
    
    args = parser.parse_args()
    
    root = Path(args.input_dir)
    out_root = args.output_dir
    
    splits = [
        ("night/train", "night_train"),
        ("night/val", "night_val"),
        ("night/test", "night_test"),
        ("day/train", "day_train"),
        ("day/val", "day_val"),
        ("day/test", "day_test")
    ]
    
    print("="*60)
    print("COCO DATASET PREPARATION")
    print("="*60)
    print(f"Input: {root}")
    print(f"Output: {out_root}")
    print(f"Sample ratio: {args.sample_ratio}")
    print(f"Box size: {args.box_px}px")
    
    all_coco = []
    for subdir, name in splits:
        split_dir = root / subdir
        if split_dir.exists():
            coco_file = process_split(split_dir, name, out_root, 
                                     args.box_px, args.sample_ratio)
            all_coco.append(coco_file)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Created {len(all_coco)} COCO annotation files:")
    for f in all_coco:
        print(f"  - {f}")


if __name__ == "__main__":
    main()