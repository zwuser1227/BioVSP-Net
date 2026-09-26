"""
CAM-YOLO End-to-End Training with CLIP Text Priors
优化版本：减少内存占用，使用更轻量级CLIP模型
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.cuda.empty_cache()

from ultralytics import YOLO

def train_cam_yolo():
    model = YOLO('ultralytics/cfg/models/CAM_YOLO/CAM_YOLO.yaml')

    results = model.train(
        data='ultralytics/cfg/datasets/NWHU.yaml',
        epochs=1000,
        imgsz=640,
        batch=2,
        device=1,
        project='runs/cam_yolo_no_DFRM_CAMfusion',
        name='train',
        exist_ok=True,
        pretrained=False,
        optimizer='SGD',
        lr0=0.01,
        weight_decay=0.001,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        close_mosaic=10,
        amp=True,
        cache=False,
        verbose=True,
    )

    return results

if __name__ == '__main__':
    results = train_cam_yolo()
    print("\nTraining completed!")
    print(f"Best weights saved to: runs/cam_yolo_no_DFRM_CAMfusion/train/weights/best.pt")
