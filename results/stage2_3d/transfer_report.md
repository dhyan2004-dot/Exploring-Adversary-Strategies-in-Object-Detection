# Transfer matrix (pooled over folds, no corruption at test)

Rows: detector the patch was TRAINED on. Columns: detector it is TESTED on. **Bold** = white-box (same detector). Cell: ASR (margin over best control [95% photo CI]).


## Threshold 0.5

| tested on | yolov8s (n=143, 13 photos) | yolov8m (n=178, 16 photos) | yolo11s (n=135, 14 photos) | rtdetr (n=158, 17 photos) |
|---|---|---|---|---|
| *best control* | 0.161 | 0.213 | 0.378 | 0.177 |
| **cpa_none** |  |  |  |  |
| &nbsp;&nbsp;trained on yolov8s | **0.315 (+0.154 [+0.00, +0.38])** | 0.348 (+0.135 [+0.07, +0.20]) | 0.496 (+0.119 [+0.02, +0.24]) | 0.443 (+0.266 [+0.14, +0.43]) |
| &nbsp;&nbsp;trained on yolov8m | 0.224 (+0.063 [-0.05, +0.23]) | **0.376 (+0.163 [+0.06, +0.29])** | 0.496 (+0.119 [+0.01, +0.26]) | 0.304 (+0.127 [+0.06, +0.22]) |
| &nbsp;&nbsp;trained on yolo11s | 0.175 (+0.014 [-0.05, +0.14]) | 0.365 (+0.152 [+0.07, +0.24]) | **0.570 (+0.193 [+0.08, +0.33])** | 0.392 (+0.215 [+0.13, +0.33]) |
| &nbsp;&nbsp;trained on rtdetr | 0.140 (-0.021 [-0.09, +0.05]) | 0.264 (+0.051 [-0.01, +0.13]) | 0.378 (+0.000 [-0.12, +0.15]) | **0.449 (+0.272 [+0.12, +0.45])** |
| **cpa_ls** |  |  |  |  |
| &nbsp;&nbsp;trained on yolov8s | **0.315 (+0.154 [+0.03, +0.35])** | 0.326 (+0.112 [+0.04, +0.18]) | 0.504 (+0.126 [+0.02, +0.26]) | 0.418 (+0.241 [+0.13, +0.38]) |
| &nbsp;&nbsp;trained on yolov8m | 0.189 (+0.028 [-0.06, +0.16]) | **0.348 (+0.135 [+0.04, +0.24])** | 0.474 (+0.096 [+0.00, +0.23]) | 0.335 (+0.158 [+0.08, +0.24]) |
| &nbsp;&nbsp;trained on yolo11s | 0.161 (+0.000 [-0.06, +0.08]) | 0.360 (+0.146 [+0.07, +0.23]) | **0.607 (+0.230 [+0.09, +0.39])** | 0.418 (+0.241 [+0.09, +0.43]) |
| &nbsp;&nbsp;trained on rtdetr | 0.147 (-0.014 [-0.07, +0.03]) | 0.253 (+0.039 [-0.00, +0.11]) | 0.393 (+0.015 [-0.12, +0.17]) | **0.443 (+0.266 [+0.12, +0.45])** |
| **npa_stop** |  |  |  |  |
| &nbsp;&nbsp;trained on yolov8s | **0.252 (+0.091 [+0.04, +0.17])** | 0.253 (+0.039 [-0.00, +0.11]) | 0.533 (+0.156 [+0.04, +0.30]) | 0.304 (+0.127 [+0.05, +0.23]) |
| &nbsp;&nbsp;trained on yolov8m | 0.203 (+0.042 [-0.04, +0.13]) | **0.337 (+0.124 [+0.07, +0.21])** | 0.511 (+0.133 [+0.02, +0.26]) | 0.272 (+0.095 [+0.04, +0.18]) |
| &nbsp;&nbsp;trained on yolo11s | 0.217 (+0.056 [+0.00, +0.13]) | 0.258 (+0.045 [-0.00, +0.12]) | **0.548 (+0.170 [+0.06, +0.30])** | 0.272 (+0.095 [+0.02, +0.19]) |
| &nbsp;&nbsp;trained on rtdetr | 0.217 (+0.056 [+0.02, +0.11]) | 0.309 (+0.096 [+0.05, +0.16]) | 0.563 (+0.185 [+0.08, +0.31]) | **0.329 (+0.152 [+0.07, +0.25])** |

**Average margin over control** (white-box = diagonal, black-box = all off-diagonal pairs)

| attack | white-box | black-box | black-box / white-box |
|---|---|---|---|
| cpa_none | +0.195 | +0.103 (12 pairs) | 0.53 |
| cpa_ls | +0.196 | +0.099 (12 pairs) | 0.51 |
| npa_stop | +0.134 | +0.094 (12 pairs) | 0.70 |

### CPA claim: cpa_ls - cpa_none (positive = cpa_ls transfers better)

| tested on | black-box surrogates | diff [95% CI] | photos that differ | p | Holm p |
|---|---|---|---|---|---|
| yolov8s | yolov8m, yolo11s, rtdetr | -0.014 [-0.04, +0.00] | 9 | 0.148 | 0.742 |
| yolov8m | yolov8s, yolo11s, rtdetr | -0.013 [-0.05, +0.02] | 10 | 0.553 | 1 |
| yolo11s | yolov8s, yolov8m, rtdetr | +0.000 [-0.01, +0.01] | 5 | 1 | 1 |
| rtdetr | yolov8s, yolov8m, yolo11s | +0.011 [-0.04, +0.06] | 9 | 0.945 | 1 |
| ALL (pooled) | all off-diagonal | -0.004 [-0.02, +0.01] | 17 | 0.73 | 1 |

### npa_stop - cpa_none (positive = npa_stop transfers better)

| tested on | black-box surrogates | diff [95% CI] | photos that differ | p | Holm p |
|---|---|---|---|---|---|
| yolov8s | yolov8m, yolo11s, rtdetr | +0.033 [-0.01, +0.06] | 12 | 0.226 | 0.677 |
| yolov8m | yolov8s, yolo11s, rtdetr | -0.052 [-0.10, -0.01] | 10 | 0.082 | 0.328 |
| yolo11s | yolov8s, yolov8m, rtdetr | +0.079 [-0.01, +0.17] | 7 | 0.234 | 0.677 |
| rtdetr | yolov8s, yolov8m, yolo11s | -0.097 [-0.15, -0.05] | 13 | 0.00146 | 0.00732 |
| ALL (pooled) | all off-diagonal | -0.015 [-0.05, +0.02] | 17 | 0.541 | 0.677 |

## Threshold 0.25

| tested on | yolov8s (n=185, 17 photos) | yolov8m (n=221, 19 photos) | yolo11s (n=188, 18 photos) | rtdetr (n=244, 21 photos) |
|---|---|---|---|---|
| *best control* | 0.081 | 0.190 | 0.335 | 0.164 |
| **cpa_none** |  |  |  |  |
| &nbsp;&nbsp;trained on yolov8s | **0.200 (+0.119 [+0.02, +0.19])** | 0.281 (+0.090 [+0.04, +0.14]) | 0.463 (+0.128 [+0.06, +0.23]) | 0.381 (+0.217 [+0.12, +0.32]) |
| &nbsp;&nbsp;trained on yolov8m | 0.146 (+0.065 [+0.00, +0.11]) | **0.267 (+0.077 [+0.03, +0.14])** | 0.468 (+0.133 [+0.04, +0.25]) | 0.307 (+0.143 [+0.07, +0.24]) |
| &nbsp;&nbsp;trained on yolo11s | 0.141 (+0.059 [-0.01, +0.11]) | 0.285 (+0.095 [+0.04, +0.17]) | **0.527 (+0.191 [+0.08, +0.31])** | 0.344 (+0.180 [+0.09, +0.27]) |
| &nbsp;&nbsp;trained on rtdetr | 0.119 (+0.038 [-0.04, +0.09]) | 0.186 (-0.005 [-0.04, +0.03]) | 0.372 (+0.037 [-0.06, +0.13]) | **0.398 (+0.234 [+0.11, +0.35])** |
| **cpa_ls** |  |  |  |  |
| &nbsp;&nbsp;trained on yolov8s | **0.222 (+0.141 [+0.04, +0.22])** | 0.262 (+0.072 [+0.03, +0.13]) | 0.495 (+0.160 [+0.08, +0.25]) | 0.402 (+0.238 [+0.12, +0.35]) |
| &nbsp;&nbsp;trained on yolov8m | 0.151 (+0.070 [-0.01, +0.11]) | **0.285 (+0.095 [+0.05, +0.15])** | 0.457 (+0.122 [+0.02, +0.26]) | 0.270 (+0.107 [+0.03, +0.20]) |
| &nbsp;&nbsp;trained on yolo11s | 0.162 (+0.081 [+0.01, +0.14]) | 0.285 (+0.095 [+0.04, +0.14]) | **0.548 (+0.213 [+0.11, +0.33])** | 0.332 (+0.168 [+0.08, +0.26]) |
| &nbsp;&nbsp;trained on rtdetr | 0.108 (+0.027 [-0.05, +0.09]) | 0.204 (+0.014 [-0.02, +0.07]) | 0.399 (+0.064 [-0.01, +0.16]) | **0.373 (+0.209 [+0.10, +0.31])** |
| **npa_stop** |  |  |  |  |
| &nbsp;&nbsp;trained on yolov8s | **0.173 (+0.092 [+0.01, +0.17])** | 0.208 (+0.018 [-0.03, +0.09]) | 0.452 (+0.117 [+0.02, +0.23]) | 0.225 (+0.061 [-0.01, +0.15]) |
| &nbsp;&nbsp;trained on yolov8m | 0.195 (+0.114 [+0.02, +0.18]) | **0.262 (+0.072 [+0.02, +0.16])** | 0.505 (+0.170 [+0.07, +0.28]) | 0.221 (+0.057 [+0.01, +0.10]) |
| &nbsp;&nbsp;trained on yolo11s | 0.162 (+0.081 [+0.00, +0.16]) | 0.204 (+0.014 [-0.04, +0.09]) | **0.457 (+0.122 [+0.04, +0.22])** | 0.250 (+0.086 [+0.02, +0.16]) |
| &nbsp;&nbsp;trained on rtdetr | 0.157 (+0.076 [-0.01, +0.16]) | 0.240 (+0.050 [+0.01, +0.12]) | 0.489 (+0.154 [+0.07, +0.25]) | **0.270 (+0.107 [+0.03, +0.20])** |

**Average margin over control** (white-box = diagonal, black-box = all off-diagonal pairs)

| attack | white-box | black-box | black-box / white-box |
|---|---|---|---|
| cpa_none | +0.155 | +0.099 (12 pairs) | 0.63 |
| cpa_ls | +0.164 | +0.101 (12 pairs) | 0.62 |
| npa_stop | +0.098 | +0.083 (12 pairs) | 0.85 |

### CPA claim: cpa_ls - cpa_none (positive = cpa_ls transfers better)

| tested on | black-box surrogates | diff [95% CI] | photos that differ | p | Holm p |
|---|---|---|---|---|---|
| yolov8s | yolov8m, yolo11s, rtdetr | +0.005 [-0.02, +0.03] | 10 | 0.736 | 1 |
| yolov8m | yolov8s, yolo11s, rtdetr | +0.000 [-0.02, +0.02] | 11 | 1 | 1 |
| yolo11s | yolov8s, yolov8m, rtdetr | +0.016 [-0.00, +0.04] | 8 | 0.18 | 0.898 |
| rtdetr | yolov8s, yolov8m, yolo11s | -0.010 [-0.05, +0.03] | 11 | 0.729 | 1 |
| ALL (pooled) | all off-diagonal | +0.002 [-0.01, +0.02] | 17 | 0.87 | 1 |

### npa_stop - cpa_none (positive = npa_stop transfers better)

| tested on | black-box surrogates | diff [95% CI] | photos that differ | p | Holm p |
|---|---|---|---|---|---|
| yolov8s | yolov8m, yolo11s, rtdetr | +0.036 [-0.01, +0.09] | 12 | 0.182 | 0.729 |
| yolov8m | yolov8s, yolo11s, rtdetr | -0.033 [-0.07, +0.02] | 14 | 0.233 | 0.729 |
| yolo11s | yolov8s, yolov8m, rtdetr | +0.048 [-0.03, +0.12] | 7 | 0.281 | 0.729 |
| rtdetr | yolov8s, yolov8m, yolo11s | -0.112 [-0.17, -0.07] | 14 | 0.000122 | 0.00061 |
| ALL (pooled) | all off-diagonal | -0.023 [-0.06, +0.01] | 21 | 0.277 | 0.729 |
