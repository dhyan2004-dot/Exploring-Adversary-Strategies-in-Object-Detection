# Detector development runs (YOLOv8s, COCO initialisation)

| run | validation data | settings | val mAP50 (best epoch / epochs) | official test mAP50 |
|---|---|---|---|---|
| initial baseline | official Validation, 16 images | 640 px, batch 32 | 0.555 (11 / 31) | - |
| file-level dev split | 12 % of training files | 960 px, rare classes oversampled | 0.86 | 0.20 |
| photo-level dev split | 11 held-out photos | 960 px, oversampled | 0.255 (2 / 42) | 0.480 |
| + frozen backbone | same | first 10 layers frozen | 0.261 (7 / 17) | - |
| no oversampling | same | - | 0.257 (3 / 13) | - |
| frozen, no oversampling | same | - | 0.269 (3 / 13) | - |
| final, per CV fold | held-out fold (16-20 photos) | 960 px, frozen, oversampled, <=15 epochs | 0.195 ± 0.037 | 0.436 ± 0.028 |
