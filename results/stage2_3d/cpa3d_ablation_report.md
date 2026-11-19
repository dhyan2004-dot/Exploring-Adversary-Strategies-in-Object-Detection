# 3-D CPA ablations (pooled over folds)

Cells: ASR (margin over the best same-size control [95% photo-level CI]). ASR = share of tanks detected before (conf >= thr) that disappear.


## Threshold 0.5

### 1. Patch size (no corruption at test)

| sheet | texture | surrogate | rtdetr | yolo11s | yolov8m |
|---|---|---|---|---|---|
| 1.0 m | cpa_none | 0.315 (+0.154 [+0.00, +0.38]) | 0.443 (+0.266 [+0.14, +0.43]) | 0.496 (+0.119 [+0.02, +0.24]) | 0.348 (+0.135 [+0.07, +0.20]) |
| 1.0 m | cpa_ls | 0.315 (+0.154 [+0.03, +0.35]) | 0.418 (+0.241 [+0.13, +0.38]) | 0.504 (+0.126 [+0.02, +0.26]) | 0.326 (+0.112 [+0.04, +0.18]) |
| 1.0 m | *best control* | 0.161 | 0.177 | 0.378 | 0.213 |
| 1.5 m | cpa_none_1p5 | 0.421 (+0.105 [-0.04, +0.30]) | 0.580 (+0.304 [+0.10, +0.54]) | 0.603 (-0.026 [-0.15, +0.12]) | 0.471 (+0.065 [-0.11, +0.28]) |
| 1.5 m | cpa_ls_1p5 | 0.436 (+0.120 [-0.02, +0.34]) | 0.551 (+0.275 [+0.05, +0.52]) | 0.569 (-0.060 [-0.19, +0.10]) | 0.619 (+0.213 [+0.03, +0.39]) |
| 1.5 m | *best control* | 0.316 | 0.275 | 0.629 | 0.406 |
| 2.0 m | cpa_none_2m | 0.677 (+0.165 [+0.05, +0.32]) | 0.572 (+0.174 [+0.02, +0.36]) | 0.629 (-0.095 [-0.29, +0.07]) | 0.613 (+0.032 [-0.16, +0.26]) |
| 2.0 m | cpa_ls_2m | 0.677 (+0.165 [+0.00, +0.31]) | 0.587 (+0.188 [+0.03, +0.37]) | 0.716 (-0.009 [-0.11, +0.09]) | 0.671 (+0.090 [-0.18, +0.32]) |
| 2.0 m | *best control* | 0.511 | 0.399 | 0.724 | 0.581 |

**cpa_ls minus cpa_none at each size** (positive = corruption training helps)

| detector | sheet | condition | diff [95% CI] | photos that differ | p | Holm p |
|---|---|---|---|---|---|---|
| surrogate | 1.0 m | none | +0.000 [-0.05, +0.05] | 9 | 1 | 1 |
| surrogate | 1.0 m | corr. avg | +0.012 [-0.01, +0.03] | 11 | 0.463 | 1 |
| surrogate | 1.5 m | none | +0.015 [-0.04, +0.10] | 5 | 0.812 | 1 |
| surrogate | 1.5 m | corr. avg | -0.002 [-0.03, +0.02] | 5 | 1 | 1 |
| surrogate | 2.0 m | none | +0.000 [-0.16, +0.15] | 5 | 1 | 1 |
| surrogate | 2.0 m | corr. avg | +0.006 [-0.14, +0.14] | 8 | 0.977 | 1 |
| rtdetr | 1.0 m | none | -0.025 [-0.05, +0.00] | 3 | 0.25 | 1 |
| rtdetr | 1.0 m | corr. avg | -0.009 [-0.03, +0.01] | 10 | 0.592 | 1 |
| rtdetr | 1.5 m | none | -0.029 [-0.08, +0.00] | 2 | 0.5 | 1 |
| rtdetr | 1.5 m | corr. avg | -0.018 [-0.05, +0.01] | 7 | 0.328 | 1 |
| rtdetr | 2.0 m | none | +0.014 [-0.02, +0.07] | 4 | 0.75 | 1 |
| rtdetr | 2.0 m | corr. avg | +0.033 [-0.01, +0.10] | 4 | 0.375 | 1 |
| yolo11s | 1.0 m | none | +0.007 [-0.03, +0.06] | 3 | 1 | 1 |
| yolo11s | 1.0 m | corr. avg | -0.009 [-0.03, +0.01] | 8 | 0.492 | 1 |
| yolo11s | 1.5 m | none | -0.034 [-0.08, +0.00] | 2 | 0.5 | 1 |
| yolo11s | 1.5 m | corr. avg | -0.006 [-0.04, +0.03] | 6 | 0.812 | 1 |
| yolo11s | 2.0 m | none | +0.086 [+0.00, +0.23] | 3 | 0.25 | 1 |
| yolo11s | 2.0 m | corr. avg | +0.060 [+0.01, +0.16] | 5 | 0.125 | 0.75 |
| yolov8m | 1.0 m | none | -0.022 [-0.10, +0.03] | 6 | 0.625 | 1 |
| yolov8m | 1.0 m | corr. avg | -0.013 [-0.04, +0.02] | 11 | 0.472 | 1 |
| yolov8m | 1.5 m | none | +0.148 [+0.05, +0.25] | 7 | 0.0469 | 0.234 |
| yolov8m | 1.5 m | corr. avg | +0.085 [+0.02, +0.16] | 8 | 0.0312 | 0.188 |
| yolov8m | 2.0 m | none | +0.058 [-0.13, +0.27] | 7 | 0.719 | 1 |
| yolov8m | 2.0 m | corr. avg | +0.097 [-0.07, +0.27] | 8 | 0.414 | 1 |

### 2. Seed variance (1 m, no corruption at test)

| texture | surrogate | rtdetr | yolo11s | yolov8m |
|---|---|---|---|---|
| cpa_none | 0.315 (+0.154 [+0.00, +0.38]) | 0.443 (+0.266 [+0.14, +0.43]) | 0.496 (+0.119 [+0.02, +0.24]) | 0.348 (+0.135 [+0.07, +0.20]) |
| cpa_none_s1 | 0.266 (+0.105 [+0.01, +0.25]) | 0.386 (+0.209 [+0.12, +0.30]) | 0.541 (+0.163 [+0.06, +0.29]) | 0.287 (+0.073 [+0.03, +0.13]) |
| cpa_ls | 0.315 (+0.154 [+0.03, +0.35]) | 0.418 (+0.241 [+0.13, +0.38]) | 0.504 (+0.126 [+0.02, +0.26]) | 0.326 (+0.112 [+0.04, +0.18]) |
| cpa_ls_s1 | 0.301 (+0.140 [+0.06, +0.26]) | 0.386 (+0.209 [+0.11, +0.31]) | 0.563 (+0.185 [+0.07, +0.32]) | 0.326 (+0.112 [+0.06, +0.20]) |

| detector | comparison | condition | diff [95% CI] | photos that differ | p | Holm p |
|---|---|---|---|---|---|---|
| surrogate | seed noise: none s1 - s0 | none | -0.049 [-0.14, +0.03] | 7 | 0.5 | 1 |
| surrogate | seed noise: none s1 - s0 | corr. avg | -0.021 [-0.08, +0.03] | 9 | 0.727 | 1 |
| surrogate | seed noise: ls s1 - s0 | none | -0.014 [-0.10, +0.06] | 7 | 0.922 | 1 |
| surrogate | seed noise: ls s1 - s0 | corr. avg | -0.007 [-0.07, +0.05] | 11 | 0.923 | 1 |
| surrogate | ls - none, seed-averaged | none | +0.017 [-0.03, +0.06] | 9 | 0.555 | 1 |
| surrogate | ls - none, seed-averaged | corr. avg | +0.019 [-0.00, +0.03] | 11 | 0.181 | 1 |
| rtdetr | seed noise: none s1 - s0 | none | -0.057 [-0.16, +0.03] | 9 | 0.359 | 1 |
| rtdetr | seed noise: none s1 - s0 | corr. avg | -0.017 [-0.06, +0.02] | 9 | 0.516 | 1 |
| rtdetr | seed noise: ls s1 - s0 | none | -0.032 [-0.12, +0.04] | 6 | 0.531 | 1 |
| rtdetr | seed noise: ls s1 - s0 | corr. avg | -0.033 [-0.08, +0.01] | 13 | 0.15 | 0.751 |
| rtdetr | ls - none, seed-averaged | none | -0.013 [-0.03, +0.00] | 8 | 0.289 | 1 |
| rtdetr | ls - none, seed-averaged | corr. avg | -0.017 [-0.03, +0.00] | 12 | 0.085 | 0.51 |
| yolo11s | seed noise: none s1 - s0 | none | +0.044 [+0.00, +0.10] | 5 | 0.25 | 1 |
| yolo11s | seed noise: none s1 - s0 | corr. avg | +0.013 [-0.03, +0.05] | 10 | 0.59 | 1 |
| yolo11s | seed noise: ls s1 - s0 | none | +0.059 [+0.00, +0.10] | 3 | 0.25 | 1 |
| yolo11s | seed noise: ls s1 - s0 | corr. avg | +0.017 [-0.01, +0.04] | 10 | 0.379 | 1 |
| yolo11s | ls - none, seed-averaged | none | +0.015 [-0.01, +0.05] | 5 | 0.5 | 1 |
| yolo11s | ls - none, seed-averaged | corr. avg | -0.007 [-0.03, +0.01] | 9 | 0.566 | 1 |
| yolov8m | seed noise: none s1 - s0 | none | -0.062 [-0.10, -0.01] | 8 | 0.0938 | 0.469 |
| yolov8m | seed noise: none s1 - s0 | corr. avg | -0.063 [-0.10, -0.02] | 11 | 0.0166 | 0.0996 |
| yolov8m | seed noise: ls s1 - s0 | none | +0.000 [-0.08, +0.11] | 7 | 1 | 1 |
| yolov8m | seed noise: ls s1 - s0 | corr. avg | -0.017 [-0.06, +0.03] | 9 | 0.68 | 1 |
| yolov8m | ls - none, seed-averaged | none | +0.008 [-0.02, +0.04] | 4 | 0.875 | 1 |
| yolov8m | ls - none, seed-averaged | corr. avg | +0.011 [-0.01, +0.03] | 12 | 0.547 | 1 |

### 3. Corruption strength during training (1 m)

| texture | surrogate none / corr. avg | rtdetr none / corr. avg | yolo11s none / corr. avg | yolov8m none / corr. avg |
|---|---|---|---|---|
| cpa_none (p=0) | 0.315 / 0.334 | 0.443 / 0.522 | 0.496 / 0.661 | 0.348 / 0.449 |
| cpa_ls (p=0.5) | 0.315 / 0.346 | 0.418 / 0.513 | 0.504 / 0.652 | 0.326 / 0.437 |
| cpa_ls_p1 (p=1.0) | 0.294 / 0.322 | 0.437 / 0.514 | 0.481 / 0.650 | 0.326 / 0.431 |

| detector | comparison | condition | diff [95% CI] | photos that differ | p | Holm p |
|---|---|---|---|---|---|---|
| surrogate | ls p=1.0 - none | none | -0.021 [-0.09, +0.04] | 5 | 0.688 | 1 |
| surrogate | ls p=1.0 - none | corr. avg | -0.012 [-0.04, +0.02] | 10 | 0.564 | 1 |
| surrogate | ls p=1.0 - ls p=0.5 | none | -0.021 [-0.07, +0.02] | 6 | 0.531 | 1 |
| surrogate | ls p=1.0 - ls p=0.5 | corr. avg | -0.024 [-0.06, +0.01] | 7 | 0.312 | 1 |
| rtdetr | ls p=1.0 - none | none | -0.006 [-0.06, +0.04] | 6 | 1 | 1 |
| rtdetr | ls p=1.0 - none | corr. avg | -0.008 [-0.05, +0.03] | 11 | 0.818 | 1 |
| rtdetr | ls p=1.0 - ls p=0.5 | none | +0.019 [-0.02, +0.05] | 6 | 0.531 | 1 |
| rtdetr | ls p=1.0 - ls p=0.5 | corr. avg | +0.002 [-0.03, +0.04] | 11 | 1 | 1 |
| yolo11s | ls p=1.0 - none | none | -0.015 [-0.09, +0.06] | 3 | 0.75 | 1 |
| yolo11s | ls p=1.0 - none | corr. avg | -0.011 [-0.03, +0.01] | 5 | 0.562 | 1 |
| yolo11s | ls p=1.0 - ls p=0.5 | none | -0.022 [-0.07, +0.01] | 4 | 0.5 | 1 |
| yolo11s | ls p=1.0 - ls p=0.5 | corr. avg | -0.002 [-0.02, +0.01] | 7 | 1 | 1 |
| yolov8m | ls p=1.0 - none | none | -0.022 [-0.08, +0.03] | 7 | 0.734 | 1 |
| yolov8m | ls p=1.0 - none | corr. avg | -0.018 [-0.04, +0.01] | 9 | 0.281 | 1 |
| yolov8m | ls p=1.0 - ls p=0.5 | none | +0.000 [-0.06, +0.10] | 4 | 1 | 1 |
| yolov8m | ls p=1.0 - ls p=0.5 | corr. avg | -0.006 [-0.03, +0.03] | 10 | 0.783 | 1 |

## Threshold 0.25

### 1. Patch size (no corruption at test)

| sheet | texture | surrogate | rtdetr | yolo11s | yolov8m |
|---|---|---|---|---|---|
| 1.0 m | cpa_none | 0.200 (+0.119 [+0.02, +0.19]) | 0.381 (+0.217 [+0.12, +0.32]) | 0.463 (+0.128 [+0.06, +0.23]) | 0.281 (+0.090 [+0.04, +0.14]) |
| 1.0 m | cpa_ls | 0.222 (+0.141 [+0.04, +0.22]) | 0.402 (+0.238 [+0.12, +0.35]) | 0.495 (+0.160 [+0.08, +0.25]) | 0.262 (+0.072 [+0.03, +0.13]) |
| 1.0 m | *best control* | 0.081 | 0.164 | 0.335 | 0.190 |
| 1.5 m | cpa_none_1p5 | 0.337 (+0.108 [-0.06, +0.31]) | 0.462 (+0.159 [+0.03, +0.29]) | 0.484 (-0.020 [-0.15, +0.12]) | 0.396 (+0.070 [-0.09, +0.21]) |
| 1.5 m | cpa_ls_1p5 | 0.355 (+0.127 [-0.03, +0.32]) | 0.476 (+0.173 [+0.00, +0.36]) | 0.444 (-0.059 [-0.19, +0.08]) | 0.465 (+0.139 [-0.05, +0.29]) |
| 1.5 m | *best control* | 0.229 | 0.303 | 0.503 | 0.326 |
| 2.0 m | cpa_none_2m | 0.506 (+0.157 [+0.05, +0.28]) | 0.543 (+0.163 [-0.01, +0.33]) | 0.562 (-0.118 [-0.28, +0.01]) | 0.465 (+0.021 [-0.17, +0.22]) |
| 2.0 m | cpa_ls_2m | 0.512 (+0.163 [+0.03, +0.30]) | 0.558 (+0.178 [-0.01, +0.36]) | 0.582 (-0.098 [-0.19, -0.02]) | 0.599 (+0.155 [-0.16, +0.39]) |
| 2.0 m | *best control* | 0.349 | 0.380 | 0.680 | 0.444 |

**cpa_ls minus cpa_none at each size** (positive = corruption training helps)

| detector | sheet | condition | diff [95% CI] | photos that differ | p | Holm p |
|---|---|---|---|---|---|---|
| surrogate | 1.0 m | none | +0.022 [+0.01, +0.04] | 4 | 0.125 | 0.75 |
| surrogate | 1.0 m | corr. avg | +0.008 [-0.01, +0.02] | 11 | 0.501 | 1 |
| surrogate | 1.5 m | none | +0.018 [-0.02, +0.06] | 5 | 0.562 | 1 |
| surrogate | 1.5 m | corr. avg | -0.003 [-0.02, +0.02] | 7 | 0.844 | 1 |
| surrogate | 2.0 m | none | +0.006 [-0.08, +0.11] | 6 | 1 | 1 |
| surrogate | 2.0 m | corr. avg | -0.014 [-0.12, +0.07] | 10 | 0.814 | 1 |
| rtdetr | 1.0 m | none | +0.020 [-0.02, +0.07] | 6 | 0.812 | 1 |
| rtdetr | 1.0 m | corr. avg | -0.008 [-0.03, +0.01] | 13 | 0.493 | 1 |
| rtdetr | 1.5 m | none | +0.014 [-0.04, +0.08] | 6 | 0.969 | 1 |
| rtdetr | 1.5 m | corr. avg | -0.004 [-0.04, +0.03] | 9 | 0.91 | 1 |
| rtdetr | 2.0 m | none | +0.014 [-0.03, +0.05] | 5 | 0.75 | 1 |
| rtdetr | 2.0 m | corr. avg | +0.023 [-0.00, +0.05] | 7 | 0.203 | 1 |
| yolo11s | 1.0 m | none | +0.032 [-0.02, +0.07] | 8 | 0.383 | 1 |
| yolo11s | 1.0 m | corr. avg | +0.012 [-0.03, +0.04] | 10 | 0.568 | 1 |
| yolo11s | 1.5 m | none | -0.039 [-0.09, +0.01] | 7 | 0.266 | 1 |
| yolo11s | 1.5 m | corr. avg | +0.003 [-0.02, +0.04] | 7 | 0.922 | 1 |
| yolo11s | 2.0 m | none | +0.020 [-0.05, +0.11] | 5 | 0.812 | 1 |
| yolo11s | 2.0 m | corr. avg | +0.028 [-0.02, +0.11] | 8 | 0.797 | 1 |
| yolov8m | 1.0 m | none | -0.018 [-0.07, +0.02] | 5 | 0.688 | 1 |
| yolov8m | 1.0 m | corr. avg | -0.017 [-0.05, +0.00] | 11 | 0.185 | 0.738 |
| yolov8m | 1.5 m | none | +0.070 [+0.04, +0.11] | 7 | 0.0156 | 0.0938 |
| yolov8m | 1.5 m | corr. avg | +0.066 [+0.01, +0.14] | 11 | 0.0605 | 0.303 |
| yolov8m | 2.0 m | none | +0.134 [-0.09, +0.37] | 6 | 0.438 | 1 |
| yolov8m | 2.0 m | corr. avg | +0.107 [-0.06, +0.30] | 7 | 0.422 | 1 |

### 2. Seed variance (1 m, no corruption at test)

| texture | surrogate | rtdetr | yolo11s | yolov8m |
|---|---|---|---|---|
| cpa_none | 0.200 (+0.119 [+0.02, +0.19]) | 0.381 (+0.217 [+0.12, +0.32]) | 0.463 (+0.128 [+0.06, +0.23]) | 0.281 (+0.090 [+0.04, +0.14]) |
| cpa_none_s1 | 0.195 (+0.114 [+0.02, +0.19]) | 0.320 (+0.156 [+0.07, +0.25]) | 0.484 (+0.149 [+0.07, +0.24]) | 0.244 (+0.054 [+0.01, +0.11]) |
| cpa_ls | 0.222 (+0.141 [+0.04, +0.22]) | 0.402 (+0.238 [+0.12, +0.35]) | 0.495 (+0.160 [+0.08, +0.25]) | 0.262 (+0.072 [+0.03, +0.13]) |
| cpa_ls_s1 | 0.200 (+0.119 [+0.03, +0.19]) | 0.307 (+0.143 [+0.05, +0.25]) | 0.516 (+0.181 [+0.08, +0.30]) | 0.281 (+0.090 [+0.03, +0.16]) |

| detector | comparison | condition | diff [95% CI] | photos that differ | p | Holm p |
|---|---|---|---|---|---|---|
| surrogate | seed noise: none s1 - s0 | none | -0.005 [-0.04, +0.03] | 3 | 1 | 1 |
| surrogate | seed noise: none s1 - s0 | corr. avg | -0.014 [-0.05, +0.01] | 8 | 0.859 | 1 |
| surrogate | seed noise: ls s1 - s0 | none | -0.022 [-0.07, +0.01] | 3 | 0.75 | 1 |
| surrogate | seed noise: ls s1 - s0 | corr. avg | -0.030 [-0.07, +0.00] | 10 | 0.199 | 0.996 |
| surrogate | ls - none, seed-averaged | none | +0.014 [+0.01, +0.02] | 5 | 0.0625 | 0.375 |
| surrogate | ls - none, seed-averaged | corr. avg | +0.000 [-0.01, +0.01] | 12 | 1 | 1 |
| rtdetr | seed noise: none s1 - s0 | none | -0.061 [-0.13, +0.00] | 10 | 0.115 | 0.576 |
| rtdetr | seed noise: none s1 - s0 | corr. avg | -0.045 [-0.07, -0.02] | 15 | 0.00549 | 0.033 |
| rtdetr | seed noise: ls s1 - s0 | none | -0.094 [-0.18, -0.00] | 7 | 0.141 | 0.576 |
| rtdetr | seed noise: ls s1 - s0 | corr. avg | -0.038 [-0.07, +0.00] | 13 | 0.119 | 0.576 |
| rtdetr | ls - none, seed-averaged | none | +0.004 [-0.02, +0.03] | 9 | 0.879 | 1 |
| rtdetr | ls - none, seed-averaged | corr. avg | -0.005 [-0.02, +0.01] | 17 | 0.531 | 1 |
| yolo11s | seed noise: none s1 - s0 | none | +0.021 [-0.03, +0.06] | 5 | 0.625 | 1 |
| yolo11s | seed noise: none s1 - s0 | corr. avg | +0.011 [-0.02, +0.03] | 10 | 0.504 | 1 |
| yolo11s | seed noise: ls s1 - s0 | none | +0.021 [-0.02, +0.08] | 8 | 0.547 | 1 |
| yolo11s | seed noise: ls s1 - s0 | corr. avg | -0.005 [-0.03, +0.02] | 9 | 0.73 | 1 |
| yolo11s | ls - none, seed-averaged | none | +0.032 [+0.01, +0.06] | 7 | 0.0156 | 0.0938 |
| yolo11s | ls - none, seed-averaged | corr. avg | +0.004 [-0.02, +0.02] | 11 | 0.764 | 1 |
| yolov8m | seed noise: none s1 - s0 | none | -0.036 [-0.07, +0.01] | 7 | 0.203 | 1 |
| yolov8m | seed noise: none s1 - s0 | corr. avg | -0.043 [-0.07, -0.01] | 14 | 0.0298 | 0.179 |
| yolov8m | seed noise: ls s1 - s0 | none | +0.018 [-0.03, +0.09] | 7 | 0.703 | 1 |
| yolov8m | seed noise: ls s1 - s0 | corr. avg | -0.011 [-0.06, +0.05] | 13 | 0.765 | 1 |
| yolov8m | ls - none, seed-averaged | none | +0.009 [-0.00, +0.02] | 5 | 0.312 | 1 |
| yolov8m | ls - none, seed-averaged | corr. avg | -0.001 [-0.01, +0.01] | 12 | 0.937 | 1 |

### 3. Corruption strength during training (1 m)

| texture | surrogate none / corr. avg | rtdetr none / corr. avg | yolo11s none / corr. avg | yolov8m none / corr. avg |
|---|---|---|---|---|
| cpa_none (p=0) | 0.200 / 0.266 | 0.381 / 0.492 | 0.463 / 0.597 | 0.281 / 0.399 |
| cpa_ls (p=0.5) | 0.222 / 0.274 | 0.402 / 0.484 | 0.495 / 0.609 | 0.262 / 0.382 |
| cpa_ls_p1 (p=1.0) | 0.227 / 0.255 | 0.328 / 0.458 | 0.441 / 0.577 | 0.253 / 0.379 |

| detector | comparison | condition | diff [95% CI] | photos that differ | p | Holm p |
|---|---|---|---|---|---|---|
| surrogate | ls p=1.0 - none | none | +0.027 [+0.00, +0.07] | 5 | 0.25 | 1 |
| surrogate | ls p=1.0 - none | corr. avg | -0.011 [-0.03, +0.01] | 10 | 0.502 | 1 |
| surrogate | ls p=1.0 - ls p=0.5 | none | +0.005 [-0.03, +0.05] | 4 | 1 | 1 |
| surrogate | ls p=1.0 - ls p=0.5 | corr. avg | -0.019 [-0.05, +0.01] | 9 | 0.43 | 1 |
| rtdetr | ls p=1.0 - none | none | -0.053 [-0.10, -0.01] | 11 | 0.0312 | 0.125 |
| rtdetr | ls p=1.0 - none | corr. avg | -0.034 [-0.07, -0.01] | 13 | 0.0361 | 0.125 |
| rtdetr | ls p=1.0 - ls p=0.5 | none | -0.074 [-0.13, -0.02] | 10 | 0.0449 | 0.125 |
| rtdetr | ls p=1.0 - ls p=0.5 | corr. avg | -0.026 [-0.05, +0.00] | 13 | 0.104 | 0.125 |
| yolo11s | ls p=1.0 - none | none | -0.021 [-0.04, -0.01] | 4 | 0.125 | 0.281 |
| yolo11s | ls p=1.0 - none | corr. avg | -0.020 [-0.05, +0.01] | 8 | 0.398 | 0.398 |
| yolo11s | ls p=1.0 - ls p=0.5 | none | -0.053 [-0.09, -0.01] | 6 | 0.0938 | 0.281 |
| yolo11s | ls p=1.0 - ls p=0.5 | corr. avg | -0.032 [-0.05, -0.01] | 7 | 0.0625 | 0.25 |
| yolov8m | ls p=1.0 - none | none | -0.027 [-0.06, +0.02] | 10 | 0.354 | 1 |
| yolov8m | ls p=1.0 - none | corr. avg | -0.020 [-0.06, +0.02] | 12 | 0.5 | 1 |
| yolov8m | ls p=1.0 - ls p=0.5 | none | -0.009 [-0.05, +0.05] | 8 | 0.875 | 1 |
| yolov8m | ls p=1.0 - ls p=0.5 | corr. avg | -0.003 [-0.05, +0.06] | 11 | 0.923 | 1 |
