"""Fine-tune a COCO-pretrained YOLOv8/YOLO11/RT-DETR detector on the prepared dataset (Ultralytics).
    python train_yolov8.py --data <data.yaml> --model yolov8s.pt --epochs 15 --imgsz 960 --freeze 10"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

DATASET_ROOT=Path(__file__).resolve().parents[1]/"data"/"Aerial Image Dataset"
DEFAULT_DATA_YAML=DATASET_ROOT/"yolo_dataset"/"data.yaml"

def parse_args():
    p=argparse.ArgumentParser(description="Fine-tune YOLOv8 on the aerial UAS dataset.")
    p.add_argument("--data",type=str,default=str(DEFAULT_DATA_YAML),
                    help="Path to data.yaml produced by prepare_dataset.py")
    p.add_argument("--model",type=str,default="yolov8s.pt",
                    help="Pretrained checkpoint to start from: yolov8n.pt / yolov8s.pt / yolov8m.pt ...")
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--batch",type=int,default=16,
                    help="Lower this (e.g. 8 or 4) if you hit a CUDA out-of-memory error.")
    p.add_argument("--patience",type=int,default=20,
                    help="Early-stopping patience (epochs with no val improvement).")
    p.add_argument("--device",type=str,default=None,
                    help="'0' for first GPU, 'cpu' for CPU. Auto-detected if not given.")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--cache",type=str,default="False",choices=["False","True","ram","disk"],
                    help="Cache images for faster epochs after the first one. 'ram' is fastest but uses "
 "a lot of RAM for ~3000 images; 'disk' caches resized images to local disk; "
 "'True' is equivalent to 'ram'. Strongly recommended if training data isn't on "
 "a local SSD (see the 'Slow image access detected' warning).")
    p.add_argument("--freeze",type=int,default=None,
                    help="Freeze the first N layers (e.g. 10 = the whole backbone), so only the neck/head "
 "adapt. Reduces memorization when training data is small/correlated.")
    p.add_argument("--lr0",type=float,default=None,
                    help="Initial learning rate. Only passed to Ultralytics if set (otherwise its default applies).")
    p.add_argument("--init-weights",type=str,default=None,
                    help="Start from these weights instead of --model's own, e.g. aerial-pretrained DOTA weights: "
 "--model yolov8s.yaml --init-weights yolov8s-obb.pt. Every layer whose shape matches is "
 "copied (backbone + neck); the class head is trained fresh for this dataset.")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--project",type=str,default="runs/detect")
    p.add_argument("--name",type=str,default="baseline_yolov8")
    p.add_argument("--resume",action="store_true",help="Resume the last interrupted run.")
    return p.parse_args()

def main():
    args=parse_args()
    data_yaml=Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(
            f"Could not find {data_yaml}\n"
 f"Run prepare_dataset.py first to generate the cleaned dataset + data.yaml."
        )
    device=args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device=="cpu":
        print("WARNING: no CUDA GPU detected -- training on CPU will be very slow. "
 "Consider reducing --epochs/--imgsz for a quick smoke test first.")
    cache={"False":False,"True":True,"ram":"ram","disk":"disk"}[args.cache]
    if args.init_weights:
        if not args.model.endswith(".yaml"):
            raise SystemExit("--init-weights needs an architecture file as --model, e.g. --model yolov8s.yaml")
        if not Path(args.init_weights).exists():
            raise SystemExit(f"{args.init_weights} not found (this node has no internet -- copy it here)")
        model=YOLO(args.model).load(args.init_weights) # prints 'Transferred N/M items'
        print(f"Initialised {args.model} from {args.init_weights}")
    elif "rtdetr" in Path(args.model).name.lower():
        from ultralytics import RTDETR # transformer detector (transfer target)
        model=RTDETR(args.model)
    else:
        model=YOLO(args.model) # loads pretrained COCO weights, adapts head to our nc automatically
    extra={}
    if args.freeze is not None:
        extra["freeze"]=args.freeze
    if args.lr0 is not None:
        extra["lr0"]=args.lr0
    train_results=model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        workers=args.workers,
        cache=cache,
        seed=args.seed,
        project=args.project,
        name=args.name,
        resume=args.resume,
        plots=True, # saves results.png, confusion_matrix.png, PR curves, etc.
        val=True, # evaluate on the val split every epoch
        exist_ok=True,
        **extra,
    )
    run_dir=getattr(train_results,"save_dir",None) or getattr(model.trainer,"save_dir",None)
    if run_dir is None:
        run_dir=Path(args.project)/args.name
        print(f"\n(Could not read back the exact save_dir Ultralytics used -- "
 f"this is the path it *should* be, but check the training log above to confirm.)")
    run_dir=Path(run_dir)
    print("\nTraining finished.")
    print(f"Best weights: {run_dir / 'weights' / 'best.pt'}")
    print(f"Last weights: {run_dir / 'weights' / 'last.pt'}")
    print(f"Curves/plots/metrics saved under: {run_dir}")
    print("\nNext step: run evaluate_yolov8.py to get the full clean-baseline report on the test set.")

if __name__=="__main__":
    main()
