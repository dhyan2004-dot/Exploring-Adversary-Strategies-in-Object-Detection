# Full-image AP50 under attack (held-out photographs, mean over 5 folds)

| Attacked class | Images | Aircraft | Building | Military Truck | Tank | mAP50 |
|---|---|---|---|---|---|---|
| Tank (YOLOv8s) | clean | 0.310 | 0.239 | 0.052 | 0.156 | 0.189 |
| Tank (YOLOv8s) | gray | 0.316 | 0.227 | 0.053 | 0.052 | 0.162 |
| Tank (YOLOv8s) | noise | 0.316 | 0.232 | 0.053 | 0.067 | 0.167 |
| Tank (YOLOv8s) | eot | 0.297 | 0.226 | 0.054 | 0.019 | 0.149 |
| Tank (YOLOv8s) | cpa_light | 0.299 | 0.220 | 0.054 | 0.018 | 0.148 |
| Tank (YOLOv8s) | cpa_shadow | 0.302 | 0.216 | 0.055 | 0.013 | 0.147 |
| Tank (YOLOv8s) | cpa_ls | 0.306 | 0.215 | 0.055 | 0.009 | 0.146 |
| Aircraft (YOLOv8s, DOTA init.) | clean | 0.431 | 0.337 | 0.093 | 0.190 | 0.263 |
| Aircraft (YOLOv8s, DOTA init.) | gray | 0.053 | 0.337 | 0.091 | 0.188 | 0.167 |
| Aircraft (YOLOv8s, DOTA init.) | noise | 0.092 | 0.332 | 0.092 | 0.189 | 0.176 |
| Aircraft (YOLOv8s, DOTA init.) | eot | 0.009 | 0.320 | 0.088 | 0.153 | 0.143 |
| Aircraft (YOLOv8s, DOTA init.) | cpa_ls | 0.010 | 0.318 | 0.089 | 0.158 | 0.144 |
