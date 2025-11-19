# Night-Time Vehicle Detection with Rain Removal

A deep learning pipeline for robust vehicle detection in challenging night-time rainy conditions, combining state-of-the-art denoising with object detection.

## 🎯 Overview

This project addresses two critical challenges in autonomous driving perception:
a. Rain Removal: Advanced denoising using Enhanced U-Net with attention mechanisms, multi-scale feature extraction, and residual learning
Vehicle Detection: Faster R-CNN for accurate pre-detection of vehicles and light instances in low-light conditions

## Key Features

🌧️ Synthetic rain generation with realistic perspective effects  
🏗️ Multiple architectures: DnCNN, U-Net, Enhanced U-Net, ResDenoiser, SwinIR, Restormer  
🚗 Dual-class detection: Vehicles and individual light instances  
📊 Comprehensive metrics: PSNR, SSIM, mAP, Precision, Recall, F1-Score  
🔄 End-to-end pipeline: Seamless integration of denoising and detection

## 📁 Project Structure  
```
├── configs/                 # Configuration files  
├── data/                    # Dataset directory  
│   ├── raw/                 # Raw PVDN dataset  
│   ├── processed/           # COCO format annotations  
│   └── synthetic_rain/      # Generated rain datasets  
├── src/  
│   ├── data_preparation/    # Dataset processing scripts  
│   │   ├── dataset.py  
│   │   ├── prepare_coco_dataset.py  
│   │   └── generate_synthetic_rain.py  
│   ├── models/  
│   │   ├── denoising/       # DnCNN, U-Net, Enhanced U-Net, etc.  
│   │   └── detection/       # Faster R-CNN  
│   ├── training/            # Training scripts  
│   │   ├── train_enhanced_unet.py  
│   │   └── train_faster_rcnn.py  
│   ├── inference/           # Inference scripts  
│   │   ├── denoise_image.py  
│   │   ├── detect_vehicles.py  
│   │   └── pipeline.py  
│   └── utils/               # Metrics, losses, visualization  
├── models/  
│   └── saved_models/        # Trained model checkpoints  
├── results/                 # Output results  
└── requirements.txt  
```
## 🚀 Installation

```bash
git clone https://github.com/yourusername/night-vehicle-detection.git
cd night-vehicle-detection
```

# Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

# Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
// Requirements: Python 3.8+, PyTorch 2.0+, CUDA-capable GPU (16GB+ VRAM recommended)
```

## 📊 Dataset Setup

Download the PVDN dataset using Kaggle API:
```python
import kagglehub
path = kagglehub.dataset_download("saralajew/provident-vehicle-detection-at-night-pvdn")
Then move the downloaded dataset to data/raw/PVDN/.
```

### Prepare Provident Detection Dataset
```bash
# Convert to COCO format for detection
python src/data_preparation/prepare_coco_dataset.py \
    --input_dir data/raw/PVDN \
    --output_dir data/processed \
    --sample_ratio 0.1
```
### Generate Synthetic Rain Dataset for Denoising
```bash
python src/data_preparation/generate_synthetic_rain.py \
    --scenes_dir data/raw/PVDN/night/train/images \
    --output_dir data/synthetic_rain \
    --num_streaks 1500
```

## 🏋️ Model Training

### Denoising Model (Enhanced U-Net)

```bash
python src/training/train_enhanced_unet.py \                #Use Required Model Training Script
    --train_input data/synthetic_rain/train/input \
    --train_target data/synthetic_rain/train/target \
    --val_input data/synthetic_rain/val/input \
    --val_target data/synthetic_rain/val/target \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4
    --weight_decay 1e-5
# Edit configs/config.yaml for more options.
```

### Detection Model (Faster R-CNN)

```bash
python src/training/train_faster_rcnn.py \                  
    --train_json data/processed/coco/night_train.json \
    --train_img_root data/raw/PVDN/night/train/images \
    --val_json data/processed/coco/night_val.json \
    --val_img_root data/raw/PVDN/night/val/images \
    --epochs 20 \
    --batch_size 4 \
    --lr 0.005
```

## 🔍 Inference

### Single Image Denoising
```bash 
python src/inference/denoise_image.py \
    --model_path models/saved_models/enhanced_unet_best.pth \
    --image_path path/to/rainy_image.png \
    --output_path results/denoised.png \
    --model_type enhanced_unet
```

### Single Image Detection
```bash
python src/inference/detect_vehicles.py \
    --model_path models/saved_models/faster_rcnn_best.pth \
    --image_path path/to/night_image.png \
    --output_path results/detected.png \
    --score_threshold 0.5
```
### End-to-End Pipeline
```bash 
python src/inference/pipeline.py \
    --denoiser_path models/saved_models/enhanced_unet_best.pth \
    --detector_path models/saved_models/faster_rcnn_best.pth \
    --image_path path/to/rainy_night_image.png \
    --output_dir results/pipeline
```

## 📈 Results

### Denoising Performance

<table>
  <tr>
    <!-- Left column containing BOTH images vertically -->
    <th rowspan="4" style="text-align:center; vertical-align:middle;">
      <!-- Rainy Input -->
      <div style="margin-bottom: 20px;">
        <img width="300" alt="Rainy Input" 
             src="https://github.com/user-attachments/assets/91a7994a-701c-4c55-9616-1425d0e77328" />
        <div style="font-weight:bold; margin-top:5px;">Rainy Input</div>
      </div>
      <!-- Ground Truth -->
      <div>
        <img width="300" alt="Ground Truth" 
             src="https://github.com/user-attachments/assets/5eca9976-7bdf-4dd4-be9e-5b42f03a970f" />
        <div style="font-weight:bold; margin-top:5px;">Ground Truth</div>
      </div>
    </th>
    <!-- Header for model outputs -->
    <th colspan="2" style="text-align:center; font-size:16px;">Model Outputs</th>
  </tr>
  <!-- ROW 1 -->
  <tr>
    <td style="text-align:center; vertical-align:top;">
      <img width="285" alt="DnCNN" src="https://github.com/user-attachments/assets/b01121a8-cabf-40bb-bdcb-ffba5693a4fc" />
      <div style="font-weight:bold; margin-top:5px;">Denoising CNN (DnCNN)</div>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <img width="285" alt="U-Net" src="https://github.com/user-attachments/assets/35c64f60-bb06-4987-a056-2aaffc1673c4" />
      <div style="font-weight:bold; margin-top:5px;">Basic U-Net</div>
    </td>
  </tr>
  <!-- ROW 2 -->
  <tr>
    <td style="text-align:center; vertical-align:top;">
      <img width="285" alt="ResNet" src="https://github.com/user-attachments/assets/80ba1315-0dd8-4aa1-b85b-9247404862ea" />
      <div style="font-weight:bold; margin-top:5px;">ResNet Denoiser</div>
    </td>
    <td style="text-align:center; vertical-align:top;">
      <img width="260" alt="SwinIR" src="https://github.com/user-attachments/assets/77a0462f-1696-4171-92f0-a3d1e06f9a6f" /

#### Comparison against state-of-the-art rain removal models.

| Metric      | SwinIR  | Restormer | ProvRain Denoiser (Ours) |
|-------------|---------|-----------|---------------------------|
| PSNR (dB)   | 31.45   | 32.72     | 36.24                     |
| SSIM        | 0.8874  | 0.8961    | 0.941                     |
| L1 Loss     | 0.0108  | 0.0099    | 0.0096                    |
| MSE         | 0.00072 | 0.00051   | 0.000238                  |
| RMSE        | 0.0268  | 0.0226    | 0.0186                    |
| MAE         | 0.0132  | 0.0120    | 0.0111                    |
| LPIPS       | 0.1623  | 0.1389    | 0.1264                    |

##### ProvRain achieves the highest reconstruction quality across all metrics.

### Detection Pipeline Performance
#### Evaluated on downstream early-warning detection.

| Metric                    | Faster R-CNN (No Denoising) | ProvRain Pipeline (Ours) |
|---------------------------|------------------------------|----------------------------|
| Proposal Recall (%)       | 78.28                        | 88.53                      |
| Classifier Accuracy (%)   | 81.74                        | 90.68                      |
| Early-warning Success (%) | 65.92                        | 88.72                      |
| Avg FPS                   | 20                           | 14                         |

## 📄 License
This project is licensed under the MIT License - see LICENSE for details.
