# AE5510 - UAS Object Detection with YOLOv8

Baseline object detection pipeline for the AE5510 *Security of Safety-Critical Systems* course project. The project specification uses aerial UAS imagery with four classes: **Aircraft, Military Truck, Tank, and Building** and requires a clean detector baseline before subsequent adversarial-attack evaluation.

## What is included

- `prepare_dataset.py` - validates the raw dataset, converts polygon/OBB-style annotations to axis-aligned YOLO boxes when required, cleans coordinates, verifies class mapping, and writes a YOLOv8-ready dataset.
- `train_yolov8.py` - fine-tunes a pretrained **YOLOv8s** detector.
- `evaluate_yolov8.py` - evaluates a trained model on train/validation/test splits and saves detection metrics and qualitative predictions.
- `weights/best.pt` - best checkpoint from the supplied training run.
- `weights/last.pt` - final checkpoint from the supplied training run.
- `results/` - training curves, confusion matrices, validation plots, arguments, and `results.csv`.

The dataset itself is **not** included in this repository.

## Baseline training configuration

The archived run used:

- Model: YOLOv8s pretrained checkpoint
- Image size: 640
- Batch size: 32
- Seed: 42
- Device: GPU 2 in the original cluster run
- Augmentation: Ultralytics training augmentation including mosaic, random affine/scale and color augmentation, together with the augmented source dataset

See `results/train_args.yaml` for the full recorded configuration.

## Baseline validation results

The archived run contains 31 logged epochs. The best logged validation results in `results/results.csv` are:

| Metric | Value | Epoch |
|---|---:|---:|
| Precision | 0.90087 | 27 |
| Recall | 0.52270 | 8 |
| mAP@0.50 | **0.55475** | 11 |
| mAP@0.50:0.95 | **0.35007** | 11 |

These are **validation** results from the archived training run; no test-set metrics are included in this repository.

## Reproducing the pipeline

### 1. Install dependencies

Install PyTorch first for your CUDA/CPU setup, then:

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

```bash
python prepare_dataset.py --root "/path/to/Aerial Image Dataset"
```

The prepared dataset will be written under the dataset root unless `--output-dir` is supplied.

### 3. Train

```bash
python train_yolov8.py --data "/path/to/Aerial Image Dataset/yolo_dataset/data.yaml" --model yolov8s.pt --epochs 100 --batch 16 --imgsz 640
```

### 4. Evaluate

```bash
python evaluate_yolov8.py --data "/path/to/Aerial Image Dataset/yolo_dataset/data.yaml" --weights weights/best.pt --split val
```

Use `--split test` when a held-out test split is available.

## Project context

The clean detector is the baseline for the next stage of the course project: generating and evaluating physically realizable adversarial perturbations and measuring degradation relative to the clean detector.
