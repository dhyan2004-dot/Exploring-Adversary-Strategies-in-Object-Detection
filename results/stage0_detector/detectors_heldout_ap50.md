# Held-out AP50 per class, mean over 5 photo-grouped folds

| detector | Aircraft | Building | Military Truck | Tank | mAP50 | official test mAP50 |
|---|---|---|---|---|---|---|
| YOLOv8s, 960 px (surrogate) | 0.288 | 0.249 | 0.060 | 0.180 | 0.195 | 0.436 |
| YOLOv8s, 1280 px | 0.323 | 0.281 | 0.062 | 0.163 | 0.208 | 0.439 |
| YOLOv8s, DOTA init., 1280 px | 0.410 | 0.277 | 0.079 | 0.156 | 0.230 | 0.438 |
| YOLOv8m (black-box) | 0.332 | 0.265 | 0.069 | 0.199 | 0.216 | - |
| YOLO11s (black-box) | 0.341 | 0.298 | 0.089 | 0.196 | 0.231 | - |
| RT-DETR-l (black-box) | 0.383 | 0.356 | 0.099 | 0.225 | 0.266 | - |
