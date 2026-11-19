=================== cv_s1280 ===================
## 1. Detector (5-fold, photo-grouped)

**held-out photos (never trained on; also used to pick the checkpoint -> slightly optimistic)** — 5 folds
| fold | mAP50 | mAP50-95 | Aircraft AP50 | Building AP50 | Military Truck AP50 | Tank AP50 |
|---|---|---|---|---|---|---|
| 0 | 0.185 | 0.086 | 0.132 | 0.401 | 0.147 | 0.061 |
| 1 | 0.286 | 0.157 | 0.343 | 0.303 | 0.090 | 0.410 |
| 2 | 0.219 | 0.124 | 0.348 | 0.239 | 0.001 | 0.289 |
| 3 | 0.171 | 0.087 | 0.288 | 0.321 | 0.072 | 0.003 |
| 4 | 0.176 | 0.103 | 0.503 | 0.143 | 0.003 | 0.053 |
| **mean ± sd** | 0.208 ± 0.048 | 0.111 ± 0.030 | 0.323 ± 0.134 | 0.281 ± 0.096 | 0.062 ± 0.062 | 0.163 ± 0.177 |

**official Testing split (independent of all folds)** — 5 folds
| fold | mAP50 | mAP50-95 | Aircraft AP50 | Building AP50 | Military Truck AP50 | Tank AP50 |
|---|---|---|---|---|---|---|
| 0 | 0.493 | 0.250 | 0.686 | 0.682 | 0.029 | 0.574 |
| 1 | 0.453 | 0.229 | 0.706 | 0.501 | 0.067 | 0.538 |
| 2 | 0.391 | 0.203 | 0.657 | 0.481 | 0.022 | 0.405 |
| 3 | 0.462 | 0.240 | 0.637 | 0.650 | 0.047 | 0.512 |
| 4 | 0.397 | 0.233 | 0.650 | 0.549 | 0.062 | 0.327 |
| **mean ± sd** | 0.439 ± 0.044 | 0.231 ± 0.018 | 0.667 ± 0.028 | 0.573 ± 0.090 | 0.046 ± 0.020 | 0.471 ± 0.102 |



=================== cv_dota1280 ===================
## 1. Detector (5-fold, photo-grouped)

**held-out photos (never trained on; also used to pick the checkpoint -> slightly optimistic)** — 5 folds
| fold | mAP50 | mAP50-95 | Aircraft AP50 | Building AP50 | Military Truck AP50 | Tank AP50 |
|---|---|---|---|---|---|---|
| 0 | 0.225 | 0.113 | 0.184 | 0.416 | 0.228 | 0.074 |
| 1 | 0.247 | 0.126 | 0.383 | 0.225 | 0.101 | 0.277 |
| 2 | 0.175 | 0.093 | 0.418 | 0.203 | 0.010 | 0.069 |
| 3 | 0.252 | 0.138 | 0.466 | 0.259 | 0.036 | 0.249 |
| 4 | 0.253 | 0.142 | 0.599 | 0.281 | 0.019 | 0.112 |
| **mean ± sd** | 0.230 ± 0.033 | 0.122 ± 0.020 | 0.410 ± 0.151 | 0.277 ± 0.083 | 0.079 ± 0.091 | 0.156 ± 0.100 |

**official Testing split (independent of all folds)** — 5 folds
| fold | mAP50 | mAP50-95 | Aircraft AP50 | Building AP50 | Military Truck AP50 | Tank AP50 |
|---|---|---|---|---|---|---|
| 0 | 0.421 | 0.228 | 0.751 | 0.554 | 0.079 | 0.301 |
| 1 | 0.418 | 0.200 | 0.767 | 0.470 | 0.199 | 0.238 |
| 2 | 0.461 | 0.227 | 0.774 | 0.459 | 0.149 | 0.462 |
| 3 | 0.491 | 0.225 | 0.736 | 0.540 | 0.160 | 0.530 |
| 4 | 0.398 | 0.194 | 0.806 | 0.481 | 0.085 | 0.221 |
| **mean ± sd** | 0.438 ± 0.037 | 0.215 ± 0.016 | 0.767 ± 0.026 | 0.501 ± 0.043 | 0.134 ± 0.051 | 0.350 ± 0.138 |


cv_s1280 fold0: 9 epochs
cv_s1280 fold1: 7 epochs
cv_s1280 fold2: 6 epochs
cv_s1280 fold3: 8 epochs
cv_s1280 fold4: 12 epochs
cv_dota1280 fold0: 11 epochs
cv_dota1280 fold1: 6 epochs
cv_dota1280 fold2: 7 epochs
cv_dota1280 fold3: 7 epochs
cv_dota1280 fold4: 7 epochs
