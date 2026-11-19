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
                    help="Pretrained checkpoint to start from: yolov8n.pt / yolov8s.pt / yolov8m.pt")
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--batch",type=int,default=16,
                    help="Lower this if GPU memory runs out")
    p.add_argument("--patience",type=int,default=20,
                    help="Early-stopping patience (epochs with no val improvement)")
    p.add_argument("--device",type=str,default=None,
                    help="'0' for first GPU, 'cpu' for CPU. Auto-detected if not given")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--cache",type=str,default="False",choices=["False","True","ram","disk"],
                    help="Cache images for faster epochs after the first one")
    p.add_argument("--freeze",type=int,default=None,
                    help="Freeze the first N layers (10 = backbone)")
    p.add_argument("--lr0",type=float,default=None,
                    help="Initial learning rate")
    p.add_argument("--init-weights",type=str,default=None,
                    help="Start from these weights instead of --model's own, e.g. DOTA-pretrained")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--project",type=str,default="runs/detect")
    p.add_argument("--name",type=str,default="baseline_yolov8")
    p.add_argument("--resume",action="store_true",help="Resume the last interrupted run")
    return p.parse_args()

def main():
    args=parse_args()
    data_yaml=Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} not found")
    device=args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    cache={"False":False,"True":True,"ram":"ram","disk":"disk"}[args.cache]
    if args.init_weights:
        if not args.model.endswith(".yaml"):
            raise SystemExit("--init-weights requires a .yaml --model")
        if not Path(args.init_weights).exists():
            raise SystemExit(f"{args.init_weights} not found")
        model=YOLO(args.model).load(args.init_weights)
    elif "rtdetr" in Path(args.model).name.lower():
        from ultralytics import RTDETR # transformer detector (transfer target)
        model=RTDETR(args.model)
    else:
        model=YOLO(args.model)
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
    run_dir=Path(run_dir)
    print(run_dir/"weights"/"best.pt")

if __name__=="__main__":
    main()
