# Physical-Domain Adversarial Patches Against UAV Object Detectors

Course project for **AE5510 Security of Safety-Critical Systems**. A printable adversarial patch on a military
vehicle is optimised to stop a UAV object detector from reporting it, and is tested on detectors the attacker never
saw. Two published methods are re-implemented and tested on real aerial photos:

* **CPA**: Pattern Corruption-Assisted physical attack (Zhang et al., IEEE JSTARS 2024): a 3-D roof texture
  trained with light-spot / shadow corruptions to improve transferability.
* **NPA**: Naturalistic Physical Adversarial patch (Hu et al., ICCV 2021): the patch is a BigGAN image and only
  the GAN latent is optimised.

The full write-up is in [`report/AE5510_report.pdf`](report/AE5510_report.pdf).

## Main findings (5-fold photo-grouped CV, ASR at confidence 0.5, margin = ASR minus a same-size gray/noise sheet)

| Finding | Evidence |
|---|---|
| A printable 1 m sheet, always on the hull, hides tanks beyond occlusion on all 4 detectors | margin +0.12 to +0.27 (YOLOv8s surrogate; YOLOv8m, YOLO11s, RT-DETR-l black-box) |
| Patches transfer | about half of the white-box margin survives on unseen detectors (+0.10 of +0.20) |
| CPA's corruption training does not improve transfer | light+shadow minus none, 12 black-box pairs: -0.004 [-0.02, +0.01], p=0.73; also null for size, seeds, corruption strength and fixed-view training |
| NPA's latent clamp alone does not keep the patch natural | 97-99 % saturated pixels even at tau=2 |
| A naturalness-budget early stop gives photo-like patches with a real attack | 8 % saturated, margin +0.10, keeps 70 % of its effect on unseen detectors |

## Repository layout

```
src/                       stage 1: dataset, detectors, 2-D patches, cross-validation and statistics
  prepare_dataset.py         raw dataset -> YOLO format
  dataset_utils.py           source-photo grouping key
  make_group_split.py        5 photo-grouped folds
  train_yolov8.py            detector fine-tuning (YOLOv8 / YOLO11 / RT-DETR)
  evaluate_yolov8.py         detector evaluation
  patch_train.py             2-D EOT patch (+ CPA-style patch corruptions)
  corruptions.py             light / shadow / rain / mud for the 2-D stage
  patch_evaluate.py          per-instance attack evaluation with gray/noise controls
  patch_map_drop.py          full-image mAP drop under attack
  photo_stats.py             photo-level bootstrap and sign-flip tests
  cv_run.py, cv_analyze.py   per-fold pipeline and pooled report
  run_all.sh, improve.sh     stage-1 drivers (main run; detector variants)
  cpa3d/                     stage 2: 3-D CPA, NPA, camouflage, transfer
    aux_mesh.py, aux_mesh.npz  roof mesh (1,922 facets)
    render.py                  differentiable UAV-view renderer (Phong, FoV 45)
    place.py                   1 m printable sheet on the estimated hull, in-vehicle clip
    corrupt_paper.py           CPA paper corruptions (light spot, shadow, rain, mud, affine)
    generator.py               CPA texture generator
    cpa_train.py, cpa_eval.py  3-D CPA training and evaluation
    npa_train.py, camo.py      NPA (BigGAN latent), pixel baseline, AdvCaT camouflage
    biggan/                    BigGAN-128 generator code (MIT, from the NPA authors)
    cpa_analyze.py, cpa_ablation.py, cpa_diversity.py, cpa_transfer.py, npa_naturalness.py, npa_sweep.py
    run_cpa3d.sh               stage-2 driver
results/                   result tables (markdown) used in the report
report/                    LaTeX source, figures and PDF of the final report
data/                      dataset location (not included)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision            # choose the CUDA build for your machine
pip install -r requirements.txt
```

Weights (not in the repository): `yolov8s.pt`, `yolov8m.pt`, `yolo11s.pt`, `rtdetr-l.pt` are downloaded by
Ultralytics on first use; for NPA put the BigGAN-128 `G_ema.pth` from the NPA authors' `pretrained_biggan.tar`
at `src/pretrained/generators/BigGAN/G_ema.pth` (or set `BIGGAN=...`).

## Reproducing

All commands are run from `src/`. Dataset placement: see [`data/README.md`](data/README.md).

```bash
cd src
python prepare_dataset.py
python make_group_split.py --root "../data/Aerial Image Dataset/yolo_dataset" --folds 5

# stage 1: detector per fold, 2-D EOT and CPA patches, evaluation, pooled report
python cv_run.py --root "../data/Aerial Image Dataset/yolo_dataset" --out cv_runs_e80 --folds all \
  --det-args "--model yolov8s.pt --epochs 15 --imgsz 960 --batch 16 --patience 5 --freeze 10" \
  --patch-epochs 80 --seeds 0,1,2          # run_all.sh wraps the same steps (with a smoke test)
python cv_analyze.py --out cv_runs_e80

# stage 2: 3-D CPA + black-box detectors, then the extra studies
bash cpa3d/run_cpa3d.sh check
bash cpa3d/run_cpa3d.sh all && bash cpa3d/run_cpa3d.sh report
bash cpa3d/run_cpa3d.sh ablation && bash cpa3d/run_cpa3d.sh ablation_report
bash cpa3d/run_cpa3d.sh transfer && bash cpa3d/run_cpa3d.sh transfer_report
bash cpa3d/run_cpa3d.sh npa && bash cpa3d/run_cpa3d.sh npa_tau && bash cpa3d/run_cpa3d.sh npa_tau_report
bash cpa3d/run_cpa3d.sh natural && bash cpa3d/run_cpa3d.sh natural2 && bash cpa3d/run_cpa3d.sh natural2_report
bash cpa3d/run_cpa3d.sh npa_sweep && bash cpa3d/run_cpa3d.sh trnat && bash cpa3d/run_cpa3d.sh sweep_trnat_report
bash cpa3d/run_cpa3d.sh diversity && bash cpa3d/run_cpa3d.sh diversity_report
```

Every stage is resumable (finished steps are skipped). The experiments used one GPU per fold.

## References

* Y. Zhang et al., Pattern Corruption-Assisted Physical Attacks Against Object Detection in UAV Remote Sensing,
  IEEE JSTARS 17, 2024. Code: github.com/zhangyu13a/CPA
* Y.-C.-T. Hu et al., Naturalistic Physical Adversarial Patch for Object Detectors, ICCV 2021.
  Code: github.com/aiiu-lab/Naturalistic-Adversarial-Patch
* Z. Hu et al., Physically Realizable Natural-Looking Clothing Textures Evade Person Detectors via 3D Modeling
  (AdvCaT), CVPR 2023.

## License

MIT (see `LICENSE`). `src/cpa3d/biggan/` keeps its own MIT licence.
