# A Comparative Study of Robustness- and Naturalness-Aware Adversarial Patch Attacks for UAS Object Detection

AE5510 Security of Safety-Critical Systems, course project (Dhyan G, AE22B030).

Physical-domain object-disappearance attacks against a UAS object detector (classes Aircraft, Military Truck,
Tank, Building). A YOLOv8s detector is fine-tuned as the baseline, and three patch strategies are compared on
the same data and evaluation:

* **EOT**: standard adversarial patch optimised over random transformations (Athalye et al. 2018).
* **CPA**: Pattern Corruption-Assisted attack (Zhang et al., IEEE JSTARS 2024): a 3-D roof texture trained
  with light-spot and shadow corruptions to improve robustness and transferability.
* **NPA**: Naturalistic Physical Adversarial patch (Hu et al., ICCV 2021): only the latent vector of a
  pretrained BigGAN is optimised, so the patch stays photo-like. AdvCaT camouflage is tested as an alternative.

Report: [`report/AE5510_report.pdf`](report/AE5510_report.pdf)

## Dataset

The dataset cannot be uploaded here due to the course instructor's restriction.

## Repository layout

```
src/                         stage 1: data, detectors, 2-D patches, cross-validation, statistics
  prepare_dataset.py           raw dataset -> YOLO format (polygon labels -> boxes)
  dataset_utils.py             source-photo grouping key
  make_group_split.py          5 photo-grouped folds
  train_yolov8.py              detector fine-tuning (YOLOv8 / YOLO11 / RT-DETR)
  evaluate_yolov8.py           precision, recall, AP50, AP50-95 per class
  patch_train.py               2-D EOT patch (+ CPA-style patch corruptions)
  corruptions.py               light / shadow / rain / mud for the 2-D stage
  patch_evaluate.py            per-instance attack evaluation with gray / noise controls
  patch_map_drop.py            full-image AP50 under attack
  photo_stats.py               photo-level bootstrap and sign-flip tests
  cv_run.py, cv_analyze.py     per-fold pipeline and pooled report
  run_all.sh, improve.sh       stage-1 drivers (main run; detector variants)
  cpa3d/                       stage 2-3: 3-D CPA, NPA, camouflage, transfer
    aux_mesh.py, aux_mesh.npz    roof mesh (1,922 facets)
    render.py                    differentiable UAV-view renderer (perspective, Phong)
    place.py                     1 m printable sheet on the estimated hull, in-vehicle clip
    corrupt_paper.py             light spot, shadow, rain, mud, affine (CPA paper)
    generator.py                 CPA texture generator
    cpa_train.py, cpa_eval.py    3-D CPA training and evaluation
    npa_train.py, camo.py        NPA (BigGAN latent), pixel baseline, AdvCaT camouflage
    biggan/                      BigGAN-128 generator code (MIT, from the NPA authors)
    cpa_analyze.py, cpa_ablation.py, cpa_diversity.py, cpa_transfer.py, npa_naturalness.py, npa_sweep.py
    run_cpa3d.sh                 stage-2 driver
results/                     result tables (markdown) produced by the analysis scripts
  stage0_detector/             development runs, per-class P/R/AP, detector variants
  stage1_2d/                   pilot, 20- and 80-epoch Tank study, Aircraft, AP50 under attack
  stage2_3d/                   3-D CPA, ablations, diversity, transfer, NPA, camouflage
report/                      LaTeX source, figures and PDF of the report
```

## Requirements and usage

```bash
pip install -r requirements.txt          # PyTorch with CUDA, Ultralytics, OpenCV, NumPy, PyYAML, Pillow
cd src
python prepare_dataset.py
python make_group_split.py --root "../data/Aerial Image Dataset/yolo_dataset" --folds 5
python cv_run.py --root "../data/Aerial Image Dataset/yolo_dataset" --out cv_runs_e80 --folds all \
  --det-args "--model yolov8s.pt --epochs 15 --imgsz 960 --batch 16 --patience 5 --freeze 10" \
  --patch-epochs 80 --seeds 0,1,2
python cv_analyze.py --out cv_runs_e80
bash cpa3d/run_cpa3d.sh all && bash cpa3d/run_cpa3d.sh report
bash cpa3d/run_cpa3d.sh ablation && bash cpa3d/run_cpa3d.sh ablation_report
bash cpa3d/run_cpa3d.sh transfer && bash cpa3d/run_cpa3d.sh transfer_report
bash cpa3d/run_cpa3d.sh npa && bash cpa3d/run_cpa3d.sh npa_tau && bash cpa3d/run_cpa3d.sh npa_tau_report
bash cpa3d/run_cpa3d.sh natural && bash cpa3d/run_cpa3d.sh natural2 && bash cpa3d/run_cpa3d.sh natural2_report
bash cpa3d/run_cpa3d.sh npa_sweep && bash cpa3d/run_cpa3d.sh trnat && bash cpa3d/run_cpa3d.sh sweep_trnat_report
bash cpa3d/run_cpa3d.sh diversity && bash cpa3d/run_cpa3d.sh diversity_report
```

Detector weights (`yolov8s.pt`, `yolov8m.pt`, `yolo11s.pt`, `rtdetr-l.pt`) are fetched by Ultralytics; NPA uses
the BigGAN-128 `G_ema.pth` from the NPA authors (`BIGGAN=<path>`). Every stage is resumable.

## Results

ASR = share of tanks detected before patching that are not detected after. Margin = ASR of the patch minus
the ASR of a gray or noise sheet of the same size and place (the stronger of the two), i.e. the effect of the
adversarial pattern beyond occlusion. Intervals resample whole source photographs (the 3,168 training files are
augmentations of 88 photographs). Full tables: [`results/`](results).

### Stage 0: detector baseline

| Run | validation mAP50 | official test mAP50 |
|---|---|---|
| Initial baseline, official Validation (16 images), 640 px | 0.555 | - |
| File-level dev split, 960 px | 0.86 | 0.20 |
| Photo-level dev split, 960 px | 0.255 | 0.480 |
| + frozen backbone / no oversampling / both | 0.261 / 0.257 / 0.269 | - |
| Final: 5 photo-grouped folds, frozen, 960 px | 0.195 ± 0.037 | 0.436 ± 0.028 |

| Class (final, 5 folds) | held-out P | R | AP50 | official test P | R | AP50 |
|---|---|---|---|---|---|---|
| Aircraft | 0.50 | 0.32 | 0.29 | 0.65 | 0.63 | 0.64 |
| Building | 0.36 | 0.33 | 0.25 | 0.48 | 0.61 | 0.53 |
| Military Truck | 0.18 | 0.09 | 0.06 | 0.25 | 0.07 | 0.07 |
| Tank | 0.41 | 0.20 | 0.18 | 0.64 | 0.42 | 0.50 |

| Detector (held-out mAP50) | Aircraft | Building | Mil. Truck | Tank | mAP50 |
|---|---|---|---|---|---|
| YOLOv8s 960 px (surrogate) | 0.288 | 0.249 | 0.060 | 0.180 | 0.195 |
| YOLOv8s 1280 px | 0.323 | 0.281 | 0.062 | 0.163 | 0.208 |
| YOLOv8s DOTA init., 1280 px | 0.410 | 0.277 | 0.079 | 0.156 | 0.230 |
| YOLOv8m / YOLO11s / RT-DETR-l (black-box) | 0.33 / 0.34 / 0.38 | 0.27 / 0.30 / 0.36 | 0.07 / 0.09 / 0.10 | 0.20 / 0.20 / 0.23 | 0.216 / 0.231 / 0.266 |

Inference: the file-level score of 0.86 was leakage between augmented copies of one photograph; on unseen
photographs mAP50 is about 0.2 and every detector, including larger ones, sits in 0.19-0.27. Military Truck is
practically not detected (19 % correct, 30 % missed as background), so there is nothing to hide; Tank is the
attacked class (Aircraft also in stage 1). Freezing the backbone and higher resolution change little.

### Stage 1: 2-D patch

Pilot (2 photos, 244 tanks): a 0.3w / 0.5w / 0.7w patch hides 0.56 / 0.96 / 0.98, a gray square 0.33 / 0.69 /
0.96. Inference: a large square hides tanks by occlusion alone, so every later result is reported as a margin
over gray/noise, and width 0.4w is used. The pilot's apparent CPA advantage under mud (0.47 vs 0.25) came from two
scenes only and motivated photo-grouped cross-validation.

Cross-validated, 185 tanks in 17 held-out photos, confidence 0.25 (margin, corruption = avg of light/shadow/rain/mud):

| Method | 20 epochs, 1 seed: none | corruption | 80 epochs, 3 seeds: none | corruption [95 % CI] |
|---|---|---|---|---|
| EOT | +0.270 | +0.188 | +0.319 | +0.280 [+0.16, +0.40] |
| CPA light | +0.227 | +0.215 | +0.297 | +0.289 [+0.16, +0.41] |
| CPA shadow | +0.330 | +0.297 | +0.366 | +0.314 [+0.20, +0.42] |
| CPA light+shadow | +0.335 | +0.291 | +0.378 | +0.340 [+0.23, +0.45] |
| light+shadow - EOT (corruption) | +0.103, Holm p 0.79 | | +0.060, Holm p 0.21 | |

Inference: every 2-D patch hides about 0.3 more tanks than occlusion, also under rain and mud that were never
trained on. The 20-epoch CPA lead over EOT was an under-trained EOT patch; with 80 epochs it shrinks to 0.06,
within the seed-to-seed spread (up to ±0.06), and is not significant.

Aircraft (280 aircraft, 21 photos): the patch hides 0.84 but a gray square already 0.78; margin +0.06
[-0.01, +0.16] (EOT), +0.04 (CPA). Inference: thin aircraft are dominated by occlusion.

Full-image AP50 with the patch on every instance of the attacked class (held-out):

| Attacked class | Images | Aircraft | Building | Mil. Truck | Tank | mAP50 |
|---|---|---|---|---|---|---|
| Tank | clean | 0.310 | 0.239 | 0.052 | 0.156 | 0.189 |
| Tank | gray square | 0.316 | 0.227 | 0.053 | 0.052 | 0.162 |
| Tank | CPA light+shadow patch | 0.306 | 0.215 | 0.055 | **0.009** | 0.146 |
| Aircraft | clean | 0.431 | 0.337 | 0.093 | 0.190 | 0.263 |
| Aircraft | gray square | 0.053 | 0.337 | 0.091 | 0.188 | 0.167 |
| Aircraft | EOT patch | **0.009** | 0.320 | 0.088 | 0.153 | 0.143 |

Inference: the attack is class-specific; Tank AP50 drops by 94 % (gray square: 67 %), other classes are unchanged.

### Stage 2: printable 3-D CPA sheet (1 m on the hull, trained on YOLOv8s, confidence 0.5)

| Texture (margin) | YOLOv8s (white-box) | YOLOv8m | YOLO11s | RT-DETR-l |
|---|---|---|---|---|
| CPA none | +0.154 | +0.135 | +0.119 | +0.266 |
| CPA light | +0.126 | +0.174 | +0.111 | +0.247 |
| CPA shadow | +0.140 | +0.135 | +0.096 | +0.215 |
| CPA light+shadow | +0.154 | +0.112 | +0.126 | +0.241 |

Inference: a realistic 1 m sheet still hides 12-27 points more tanks than occlusion on all four detectors;
light, shadow and both are within 0.05 of none and no difference is significant. During training the share of
tanks below 0.5 rises from 0.62 to 0.76 (none) and 0.67 to 0.79 (light+shadow), i.e. both converge to the same
level. Ablations (1.5 m and 2 m sheets, a second seed, corruption probability 1.0, paper-like fixed views) give
light+shadow minus none on the black-box detectors between -0.03 and +0.15, none significant after Holm
correction; seed-to-seed variation alone reaches 0.06.

Transferability (mean margin, every attack trained on each of the 4 detectors):

| Attack | white-box | black-box (12 pairs) | kept |
|---|---|---|---|
| CPA none | +0.195 | +0.103 | 53 % |
| CPA light+shadow | +0.196 | +0.099 | 51 % |
| NPA early stop | +0.134 | +0.094 | 70 % |

Inference: about half of the effect transfers (strongly YOLO to RT-DETR, weakly in reverse). CPA's claim is not
reproduced: light+shadow minus none over all black-box pairs is -0.004 [-0.02, +0.01], p = 0.73.

### Stage 3: naturalness (mean margin over 4 detectors, saturated pixels)

| Patch | margin | saturated |
|---|---|---|
| NPA, paper setting (dog, tau 10) | +0.183 | 98.9 % |
| NPA, tau 2 | +0.133 | 97.2 % |
| NPA start image (not optimised) | +0.003 | 0.9 % |
| NPA, latent on sphere | +0.198 | 87.5 % |
| NPA + saturation penalty | +0.163 | 0 % (visibly unnatural) |
| NPA + naturalness early stop (20 %) | +0.103 | 7.8 % |
| AdvCaT camouflage, 1 m | +0.013 | 0 % |
| pixel patch (no GAN) | +0.173 | 64.8 % |

Inference: the latent clamp alone does not keep the patch natural (saturated pixels 0 -> 20 -> 68 -> 95 % over
the first epochs); stopping at a saturation budget does and keeps about half of the attack, and that patch
transfers best (70 %). Camouflage is natural but no better than a gray sheet.

## References

* Y. Zhang et al., Pattern Corruption-Assisted Physical Attacks Against Object Detection in UAV Remote Sensing,
  IEEE JSTARS 17, 2024.
* Y.-C.-T. Hu et al., Naturalistic Physical Adversarial Patch for Object Detectors, ICCV 2021.
* Z. Hu et al., Physically Realizable Natural-Looking Clothing Textures Evade Person Detectors via 3D Modeling
  (AdvCaT), CVPR 2023.
* A. Athalye et al., Synthesizing Robust Adversarial Examples, ICML 2018.

## License

MIT (see `LICENSE`). `src/cpa3d/biggan/` keeps its own MIT licence.
