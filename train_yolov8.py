"""
train_yolov8.py
----------------
Step 2 of the pipeline: fine-tune a pretrained YOLOv8 detector on the
cleaned dataset produced by prepare_dataset.py.

Run prepare_dataset.py first -- this script expects
<DATASET_ROOT>/yolo_dataset/data.yaml to already exist.

Usage (defaults are fine for a first run):
    python train_yolov8.py
    python train_yolov8.py --model yolov8s.pt --epochs 150 --batch 8
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

DEFAULT_DATA_YAML = Path("data.yaml")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 on the aerial UAS dataset.")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML),
                    help="Path to data.yaml produced by prepare_dataset.py")
    p.add_argument("--model", type=str, default="yolov8s.pt",
                    help="Pretrained checkpoint to start from: yolov8n.pt / yolov8s.pt / yolov8m.pt ...")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16,
                    help="Lower this (e.g. 8 or 4) if you hit a CUDA out-of-memory error.")
    p.add_argument("--patience", type=int, default=20,
                    help="Early-stopping patience (epochs with no val improvement).")
    p.add_argument("--device", type=str, default=None,
                    help="'0' for first GPU, 'cpu' for CPU. Auto-detected if not given.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--name", type=str, default="baseline_yolov8")
    p.add_argument("--resume", action="store_true", help="Resume the last interrupted run.")
    return p.parse_args()


def main():
    args = parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(
            f"Could not find {data_yaml}\n"
            f"Run prepare_dataset.py first to generate the cleaned dataset + data.yaml."
        )

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: no CUDA GPU detected -- training on CPU will be very slow. "
              "Consider reducing --epochs/--imgsz for a quick smoke test first.")

    model = YOLO(args.model)  # loads pretrained COCO weights, adapts head to our nc automatically

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        workers=args.workers,
        seed=args.seed,
        project=args.project,
        name=args.name,
        resume=args.resume,
        plots=True,          # saves results.png, confusion_matrix.png, PR curves, etc.
        val=True,             # evaluate on the val split every epoch
        exist_ok=True,
    )

    run_dir = Path(args.project) / args.name
    print("\nTraining finished.")
    print(f"Best weights: {run_dir / 'weights' / 'best.pt'}")
    print(f"Last weights: {run_dir / 'weights' / 'last.pt'}")
    print(f"Curves/plots/metrics saved under: {run_dir}")
    print("\nNext step: run evaluate_yolov8.py to get the full clean-baseline report on the test set.")


if __name__ == "__main__":
    main()
