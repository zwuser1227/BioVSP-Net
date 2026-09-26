import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.cuda.empty_cache()

from ultralytics import YOLO


def evaluate_model(weights_path, yaml_path='ultralytics/cfg/models/CAM_YOLO/CAM_YOLO.yaml'):
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Weights: {weights_path}")
    print(f"Model YAML: {yaml_path}")
    print("=" * 60)

    model = YOLO(yaml_path)
    print(next(model.model.parameters()).device)
    model.model = model.model.to('cuda:0')
    print(next(model.model.parameters()).device)
    model.load(weights_path)

    results = model.val(
        data='ultralytics/cfg/datasets/NWHU.yaml',
        imgsz=640,
        batch=2,
        device=1,
        split='val',
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"mAP50:     {results.box.map50:.4f}")
    print(f"mAP50-95:  {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")
    print("=" * 60)

    return results


if __name__ == '__main__':
    WEIGHTS_PATH = '/data/zw2024/pro/cam-yolo-N/runs/cam_yolo/train/weights/best.pt'

    evaluate_model(WEIGHTS_PATH)
