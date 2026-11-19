"""Evaluate a trained detector on a split (mAP50, mAP50-95, per-class AP) at its training image size.
    python evaluate_yolov8.py --data <data.yaml> --weights <best.pt> --split test"""

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

def extract_imgsz_from_ckpt(ckpt:dict):
    """Pull the training imgsz out of an Ultralytics checkpoint dict, if present."""
    try:
        train_args=ckpt.get("train_args",{})
        imgsz=train_args.get("imgsz") if isinstance(train_args,dict) else getattr(train_args,"imgsz",None)
        return int(imgsz) if imgsz is not None else None
    except Exception:
        return None

def detect_training_imgsz(weights_path:Path):
    try:
        ckpt=torch.load(weights_path,map_location="cpu",weights_only=False)
    except Exception:
        return None
    return extract_imgsz_from_ckpt(ckpt)

DATASET_ROOT=Path(__file__).resolve().parents[1]/"data"/"Aerial Image Dataset"
DEFAULT_DATA_YAML=DATASET_ROOT/"yolo_dataset"/"data.yaml"
DEFAULT_WEIGHTS=Path("runs/detect/baseline_yolov8/weights/best.pt")

def parse_args():
    p=argparse.ArgumentParser(description="Evaluate a fine-tuned YOLOv8 model (clean baseline).")
    p.add_argument("--data",type=str,default=str(DEFAULT_DATA_YAML))
    p.add_argument("--weights",type=str,default=str(DEFAULT_WEIGHTS))
    p.add_argument("--split",type=str,default="test",choices=["train","val","test"],
                    help="Which split in data.yaml to evaluate on.")
    p.add_argument("--imgsz",type=int,default=None,
                    help="Must match the imgsz the weights were trained with, or accuracy will suffer. "
 "If omitted, this script tries to auto-detect it from the checkpoint; "
 "falls back to 640 with a warning if that fails.")
    p.add_argument("--conf",type=float,default=0.001,
                    help="Confidence threshold used for computing mAP (kept low, standard practice).")
    p.add_argument("--pred-conf",type=float,default=0.25,
                    help="Confidence threshold used only for the saved qualitative prediction images.")
    p.add_argument("--device",type=str,default=None)
    p.add_argument("--project",type=str,default="runs/detect")
    p.add_argument("--name",type=str,default="baseline_eval")
    p.add_argument("--no-predictions",action="store_true",
                   help="Skip saving annotated prediction images (use for cross-validation runs).")
    p.add_argument("--out-dir",type=str,default="clean_baseline_report",
                    help="Where to write the metrics CSV/JSON summary.")
    return p.parse_args()

def main():
    args=parse_args()
    data_yaml=Path(args.data)
    weights=Path(args.weights)
    if not data_yaml.exists():
        raise SystemExit(f"Could not find {data_yaml}. Run prepare_dataset.py first.")
    if not weights.exists():
        raise SystemExit(f"Could not find weights at {weights}. Run train_yolov8.py first, "
 f"or pass --weights pointing at your best.pt")
    with open(data_yaml) as f:
        data_cfg=yaml.safe_load(f)
    if args.split not in data_cfg:
        raise SystemExit(f"data.yaml has no '{args.split}' split defined. "
 f"Available: {[k for k in ('train','val','test') if k in data_cfg]}")
    if args.imgsz is not None:
        imgsz=args.imgsz
    else:
        detected=detect_training_imgsz(weights)
        if detected is not None:
            imgsz=detected
            print(f"Auto-detected training imgsz={imgsz} from the checkpoint; using it for evaluation.")
        else:
            imgsz=640
            print("WARNING: could not auto-detect training imgsz from the checkpoint -- "
 "defaulting to 640. If you trained with a different --imgsz, evaluating at "
 "the wrong resolution will silently tank your metrics (this is exactly what "
 "happened when a 960-trained model got evaluated at the 640 default). "
 "Pass --imgsz explicitly to be safe.")
    _m=torch.load(str(weights),map_location="cpu",weights_only=False).get("model")
    if _m is not None and any(type(x).__name__=="RTDETRDecoder" for x in _m.modules()):
        from ultralytics import RTDETR # RT-DETR needs its own predictor/validator
        model=RTDETR(str(weights))
    else:
        model=YOLO(str(weights))
    print(f"Evaluating on '{args.split}' split...")
    results=model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=imgsz,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True, # saves confusion matrix, PR curve, F1 curve etc.
        save_json=False, # True makes Ultralytics try to pip-install faster-coco-eval (no internet -> ~13 min stall)
        exist_ok=True,
    )
    names=results.names # {id: name}
    box=results.box
    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    overall={
        "split":args.split,
        "precision(mean)":float(box.mp),
        "recall(mean)":float(box.mr),
        "mAP50":float(box.map50),
        "mAP50-95":float(box.map),
    }
    per_class=[]
    for idx,cls_id in enumerate(box.ap_class_index):
        per_class.append({
            "class_id":int(cls_id),
            "name":names[int(cls_id)],
            "precision":float(box.p[idx]),
            "recall":float(box.r[idx]),
            "AP50":float(box.ap50[idx]),
            "AP50-95":float(box.ap[idx]),
        })
    print("\n"+"="*70)
    print(f"CLEAN BASELINE -- {args.split.upper()} SET")
    print("="*70)
    print(f"Precision (mean): {overall['precision(mean)']:.4f}")
    print(f"Recall    (mean): {overall['recall(mean)']:.4f}")
    print(f"mAP50           : {overall['mAP50']:.4f}")
    print(f"mAP50-95        : {overall['mAP50-95']:.4f}")
    print("\nPer-class:")
    header=f"{'class':<16}{'precision':>10}{'recall':>10}{'AP50':>10}{'AP50-95':>10}"
    print(header)
    print("-"*len(header))
    for row in per_class:
        print(f"{row['name']:<16}{row['precision']:>10.4f}{row['recall']:>10.4f}"
 f"{row['AP50']:>10.4f}{row['AP50-95']:>10.4f}")
    with open(out_dir/f"{args.split}_metrics.json","w") as f:
        json.dump({"overall":overall,"per_class":per_class},f,indent=2)
    with open(out_dir/f"{args.split}_metrics.csv","w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["class_id","name","precision","recall","AP50","AP50-95"])
        writer.writeheader()
        writer.writerows(per_class)
    print(f"\nMetrics saved to: {out_dir / f'{args.split}_metrics.json'} and .csv")
    print(f"Plots (confusion matrix, PR/F1 curves) saved under: {Path(args.project) / args.name}")
    if args.no_predictions:
        return
    split_rel=data_cfg[args.split]
    images_dir=Path(data_cfg["path"])/split_rel
    print(f"\nSaving annotated qualitative predictions for images in {images_dir} ...")
    model.predict(
        source=str(images_dir),
        imgsz=imgsz,
        conf=args.pred_conf,
        device=args.device,
        save=True,
        project=args.project,
        name=f"{args.name}_predictions",
        exist_ok=True,
    )
    print(f"Annotated prediction images saved under: {Path(args.project) / (args.name + '_predictions')}")

if __name__=="__main__":
    main()
