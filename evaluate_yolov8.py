"""
evaluate_yolov8.py
-------------------
Step 3 of the pipeline: evaluate the fine-tuned detector on the held-out
TEST split and produce the "clean baseline" numbers you'll need before
starting the adversarial-patch work (CPA/NPA), plus a folder of
annotated qualitative predictions.

Run this after train_yolov8.py.

Usage:
    python evaluate_yolov8.py --weights runs/detect/baseline_yolov8/weights/best.pt
"""

import argparse
import csv
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

DEFAULT_DATA_YAML = Path("data.yaml")
DEFAULT_WEIGHTS = Path("runs/detect/baseline_yolov8/weights/best.pt")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned YOLOv8 model (clean baseline).")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML))
    p.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                    help="Which split in data.yaml to evaluate on.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001,
                    help="Confidence threshold used for computing mAP (kept low, standard practice).")
    p.add_argument("--pred-conf", type=float, default=0.25,
                    help="Confidence threshold used only for the saved qualitative prediction images.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--name", type=str, default="baseline_eval")
    p.add_argument("--out-dir", type=str, default="clean_baseline_report",
                    help="Where to write the metrics CSV/JSON summary.")
    return p.parse_args()


def main():
    args = parse_args()

    data_yaml = Path(args.data)
    weights = Path(args.weights)
    if not data_yaml.exists():
        raise SystemExit(f"Could not find {data_yaml}. Run prepare_dataset.py first.")
    if not weights.exists():
        raise SystemExit(f"Could not find weights at {weights}. Run train_yolov8.py first, "
                          f"or pass --weights pointing at your best.pt")

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    if args.split not in data_cfg:
        raise SystemExit(f"data.yaml has no '{args.split}' split defined. "
                          f"Available: {[k for k in ('train','val','test') if k in data_cfg]}")

    model = YOLO(str(weights))

    print(f"Evaluating on '{args.split}' split...")
    results = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,       # saves confusion matrix, PR curve, F1 curve etc.
        save_json=True,
        exist_ok=True,
    )

    names = results.names  # {id: name}
    box = results.box

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- overall metrics ----
    overall = {
        "split": args.split,
        "precision(mean)": float(box.mp),
        "recall(mean)": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
    }

    # ---- per-class metrics ----
    per_class = []
    for idx, cls_id in enumerate(box.ap_class_index):
        per_class.append({
            "class_id": int(cls_id),
            "name": names[int(cls_id)],
            "precision": float(box.p[idx]),
            "recall": float(box.r[idx]),
            "AP50": float(box.ap50[idx]),
            "AP50-95": float(box.ap[idx]),
        })

    # ---- print report ----
    print("\n" + "=" * 70)
    print(f"CLEAN BASELINE -- {args.split.upper()} SET")
    print("=" * 70)
    print(f"Precision (mean): {overall['precision(mean)']:.4f}")
    print(f"Recall    (mean): {overall['recall(mean)']:.4f}")
    print(f"mAP50           : {overall['mAP50']:.4f}")
    print(f"mAP50-95        : {overall['mAP50-95']:.4f}")
    print("\nPer-class:")
    header = f"{'class':<16}{'precision':>10}{'recall':>10}{'AP50':>10}{'AP50-95':>10}"
    print(header)
    print("-" * len(header))
    for row in per_class:
        print(f"{row['name']:<16}{row['precision']:>10.4f}{row['recall']:>10.4f}"
              f"{row['AP50']:>10.4f}{row['AP50-95']:>10.4f}")

    # ---- save to disk ----
    with open(out_dir / f"{args.split}_metrics.json", "w") as f:
        json.dump({"overall": overall, "per_class": per_class}, f, indent=2)

    with open(out_dir / f"{args.split}_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "name", "precision", "recall", "AP50", "AP50-95"])
        writer.writeheader()
        writer.writerows(per_class)

    print(f"\nMetrics saved to: {out_dir / f'{args.split}_metrics.json'} and .csv")
    print(f"Plots (confusion matrix, PR/F1 curves) saved under: {Path(args.project) / args.name}")

    # ---- qualitative predictions on the same split, for the report ----
    split_rel = data_cfg[args.split]
    images_dir = Path(data_cfg["path"]) / split_rel
    print(f"\nSaving annotated qualitative predictions for images in {images_dir} ...")
    model.predict(
        source=str(images_dir),
        imgsz=args.imgsz,
        conf=args.pred_conf,
        device=args.device,
        save=True,
        project=args.project,
        name=f"{args.name}_predictions",
        exist_ok=True,
    )
    print(f"Annotated prediction images saved under: {Path(args.project) / (args.name + '_predictions')}")


if __name__ == "__main__":
    main()
