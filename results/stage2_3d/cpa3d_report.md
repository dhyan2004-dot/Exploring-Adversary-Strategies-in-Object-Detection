# 3-D CPA results (pooled over folds)

Runs: detectors: surrogate (5 folds), rtdetr (5 folds), yolo11s (5 folds), yolov8m (5 folds)


## Threshold 0.5

### A. ASR, no corruption [95% photo-level CI]

| texture | surrogate (n=143, 13 photos) | rtdetr (n=158, 17 photos) | yolo11s (n=135, 14 photos) | yolov8m (n=178, 16 photos) |
|---|---|---|---|---|
| cpa_none | 0.315 [0.14, 0.55] | 0.443 [0.27, 0.67] | 0.496 [0.30, 0.68] | 0.348 [0.21, 0.50] |
| cpa_light | 0.287 [0.12, 0.52] | 0.424 [0.25, 0.64] | 0.489 [0.28, 0.70] | 0.388 [0.24, 0.54] |
| cpa_shadow | 0.301 [0.13, 0.54] | 0.392 [0.24, 0.59] | 0.474 [0.29, 0.66] | 0.348 [0.21, 0.51] |
| cpa_ls | 0.315 [0.17, 0.52] | 0.418 [0.26, 0.64] | 0.504 [0.31, 0.68] | 0.326 [0.20, 0.45] |
| *gray control* | 0.161 | 0.177 | 0.378 | 0.213 |
| *noise control* | 0.112 | 0.158 | 0.244 | 0.152 |

### B. Margin over the best control, no corruption [95% CI]

| texture | surrogate | rtdetr | yolo11s | yolov8m |
|---|---|---|---|---|
| cpa_none | +0.154 [+0.00, +0.38] | +0.266 [+0.14, +0.43] | +0.119 [+0.02, +0.24] | +0.135 [+0.07, +0.20] |
| cpa_light | +0.126 [-0.01, +0.35] | +0.247 [+0.12, +0.39] | +0.111 [-0.01, +0.27] | +0.174 [+0.09, +0.24] |
| cpa_shadow | +0.140 [-0.01, +0.37] | +0.215 [+0.11, +0.33] | +0.096 [-0.02, +0.23] | +0.135 [+0.06, +0.23] |
| cpa_ls | +0.154 [+0.03, +0.35] | +0.241 [+0.13, +0.38] | +0.126 [+0.02, +0.26] | +0.112 [+0.04, +0.18] |

### C. ASR per test condition


**surrogate**

| texture | none | light | shadow | rain | mud | corr. avg |
|---|---|---|---|---|---|---|
| cpa_none | 0.315 | 0.259 | 0.336 | 0.399 | 0.343 | 0.334 |
| cpa_light | 0.287 | 0.238 | 0.350 | 0.392 | 0.343 | 0.330 |
| cpa_shadow | 0.301 | 0.266 | 0.336 | 0.420 | 0.329 | 0.337 |
| cpa_ls | 0.315 | 0.245 | 0.357 | 0.427 | 0.357 | 0.346 |
| gray | 0.161 | 0.126 | 0.294 | 0.343 | 0.231 | 0.248 |
| noise | 0.112 | 0.077 | 0.231 | 0.252 | 0.140 | 0.175 |

**rtdetr**

| texture | none | light | shadow | rain | mud | corr. avg |
|---|---|---|---|---|---|---|
| cpa_none | 0.443 | 0.418 | 0.494 | 0.608 | 0.570 | 0.522 |
| cpa_light | 0.424 | 0.424 | 0.481 | 0.589 | 0.538 | 0.508 |
| cpa_shadow | 0.392 | 0.399 | 0.481 | 0.570 | 0.544 | 0.498 |
| cpa_ls | 0.418 | 0.418 | 0.519 | 0.576 | 0.538 | 0.513 |
| gray | 0.177 | 0.228 | 0.329 | 0.468 | 0.430 | 0.364 |
| noise | 0.158 | 0.184 | 0.278 | 0.462 | 0.418 | 0.335 |

**yolo11s**

| texture | none | light | shadow | rain | mud | corr. avg |
|---|---|---|---|---|---|---|
| cpa_none | 0.496 | 0.496 | 0.637 | 0.800 | 0.711 | 0.661 |
| cpa_light | 0.489 | 0.481 | 0.659 | 0.800 | 0.711 | 0.663 |
| cpa_shadow | 0.474 | 0.511 | 0.659 | 0.778 | 0.719 | 0.667 |
| cpa_ls | 0.504 | 0.481 | 0.622 | 0.785 | 0.719 | 0.652 |
| gray | 0.378 | 0.341 | 0.519 | 0.630 | 0.615 | 0.526 |
| noise | 0.244 | 0.237 | 0.415 | 0.504 | 0.481 | 0.409 |

**yolov8m**

| texture | none | light | shadow | rain | mud | corr. avg |
|---|---|---|---|---|---|---|
| cpa_none | 0.348 | 0.326 | 0.410 | 0.551 | 0.511 | 0.449 |
| cpa_light | 0.388 | 0.331 | 0.444 | 0.545 | 0.517 | 0.459 |
| cpa_shadow | 0.348 | 0.320 | 0.416 | 0.539 | 0.500 | 0.444 |
| cpa_ls | 0.326 | 0.315 | 0.410 | 0.517 | 0.506 | 0.437 |
| gray | 0.213 | 0.219 | 0.360 | 0.455 | 0.427 | 0.365 |
| noise | 0.152 | 0.163 | 0.275 | 0.376 | 0.354 | 0.292 |

### D. Paired tests vs `cpa_none` (difference in ASR; photos sign-flipped; Holm per detector)

| detector | texture | condition | diff [95% CI] | photos that differ | p | Holm p |
|---|---|---|---|---|---|---|
| surrogate | cpa_light | none | -0.028 [-0.08, +0.01] | 5 (min p 0.0625) | 0.312 | 1 |
| surrogate | cpa_light | corr. avg | -0.003 [-0.04, +0.03] | 11 (min p 0.000977) | 0.913 | 1 |
| surrogate | cpa_shadow | none | -0.014 [-0.03, +0.00] | 2 (min p 0.5) | 0.5 | 1 |
| surrogate | cpa_shadow | corr. avg | +0.003 [-0.02, +0.03] | 8 (min p 0.00781) | 0.898 | 1 |
| surrogate | cpa_ls | none | +0.000 [-0.05, +0.05] | 9 (min p 0.00391) | 1 | 1 |
| surrogate | cpa_ls | corr. avg | +0.012 [-0.01, +0.03] | 11 (min p 0.000977) | 0.463 | 1 |
| rtdetr | cpa_light | none | -0.019 [-0.10, +0.04] | 6 (min p 0.0312) | 0.688 | 1 |
| rtdetr | cpa_light | corr. avg | -0.014 [-0.05, +0.01] | 10 (min p 0.00195) | 0.441 | 1 |
| rtdetr | cpa_shadow | none | -0.051 [-0.12, +0.00] | 7 (min p 0.0156) | 0.156 | 0.938 |
| rtdetr | cpa_shadow | corr. avg | -0.024 [-0.06, +0.00] | 8 (min p 0.00781) | 0.242 | 1 |
| rtdetr | cpa_ls | none | -0.025 [-0.05, +0.00] | 3 (min p 0.25) | 0.25 | 1 |
| rtdetr | cpa_ls | corr. avg | -0.009 [-0.03, +0.01] | 10 (min p 0.00195) | 0.592 | 1 |
| yolo11s | cpa_light | none | -0.007 [-0.06, +0.05] | 4 (min p 0.125) | 1 | 1 |
| yolo11s | cpa_light | corr. avg | +0.002 [-0.03, +0.03] | 6 (min p 0.0312) | 1 | 1 |
| yolo11s | cpa_shadow | none | -0.022 [-0.09, +0.04] | 3 (min p 0.25) | 0.75 | 1 |
| yolo11s | cpa_shadow | corr. avg | +0.006 [-0.02, +0.03] | 6 (min p 0.0312) | 0.781 | 1 |
| yolo11s | cpa_ls | none | +0.007 [-0.03, +0.06] | 3 (min p 0.25) | 1 | 1 |
| yolo11s | cpa_ls | corr. avg | -0.009 [-0.03, +0.01] | 8 (min p 0.00781) | 0.492 | 1 |
| yolov8m | cpa_light | none | +0.039 [-0.01, +0.07] | 7 (min p 0.0156) | 0.172 | 1 |
| yolov8m | cpa_light | corr. avg | +0.010 [-0.02, +0.04] | 9 (min p 0.00391) | 0.66 | 1 |
| yolov8m | cpa_shadow | none | +0.000 [-0.06, +0.05] | 6 (min p 0.0312) | 1 | 1 |
| yolov8m | cpa_shadow | corr. avg | -0.006 [-0.03, +0.02] | 10 (min p 0.00195) | 0.787 | 1 |
| yolov8m | cpa_ls | none | -0.022 [-0.10, +0.03] | 6 (min p 0.0312) | 0.625 | 1 |
| yolov8m | cpa_ls | corr. avg | -0.013 [-0.04, +0.02] | 11 (min p 0.000977) | 0.472 | 1 |

## Threshold 0.25

### A. ASR, no corruption [95% photo-level CI]

| texture | surrogate (n=185, 17 photos) | rtdetr (n=244, 21 photos) | yolo11s (n=188, 18 photos) | yolov8m (n=221, 19 photos) |
|---|---|---|---|---|
| cpa_none | 0.200 [0.09, 0.31] | 0.381 [0.23, 0.57] | 0.463 [0.31, 0.64] | 0.281 [0.18, 0.42] |
| cpa_light | 0.216 [0.09, 0.33] | 0.377 [0.22, 0.57] | 0.479 [0.31, 0.67] | 0.258 [0.16, 0.40] |
| cpa_shadow | 0.238 [0.11, 0.36] | 0.352 [0.19, 0.55] | 0.468 [0.31, 0.65] | 0.281 [0.17, 0.42] |
| cpa_ls | 0.222 [0.10, 0.34] | 0.402 [0.23, 0.59] | 0.495 [0.33, 0.67] | 0.262 [0.17, 0.39] |
| *gray control* | 0.070 | 0.156 | 0.335 | 0.190 |
| *noise control* | 0.081 | 0.164 | 0.255 | 0.122 |

### B. Margin over the best control, no corruption [95% CI]

| texture | surrogate | rtdetr | yolo11s | yolov8m |
|---|---|---|---|---|
| cpa_none | +0.119 [+0.02, +0.19] | +0.217 [+0.12, +0.32] | +0.128 [+0.06, +0.23] | +0.090 [+0.04, +0.14] |
| cpa_light | +0.135 [+0.03, +0.21] | +0.213 [+0.11, +0.32] | +0.144 [+0.06, +0.25] | +0.068 [+0.02, +0.12] |
| cpa_shadow | +0.157 [+0.05, +0.24] | +0.189 [+0.09, +0.29] | +0.133 [+0.05, +0.24] | +0.090 [+0.03, +0.15] |
| cpa_ls | +0.141 [+0.04, +0.22] | +0.238 [+0.12, +0.35] | +0.160 [+0.08, +0.25] | +0.072 [+0.03, +0.13] |

### C. ASR per test condition


**surrogate**

| texture | none | light | shadow | rain | mud | corr. avg |
|---|---|---|---|---|---|---|
| cpa_none | 0.200 | 0.189 | 0.249 | 0.346 | 0.281 | 0.266 |
| cpa_light | 0.216 | 0.195 | 0.265 | 0.319 | 0.276 | 0.264 |
| cpa_shadow | 0.238 | 0.200 | 0.259 | 0.357 | 0.281 | 0.274 |
| cpa_ls | 0.222 | 0.211 | 0.249 | 0.341 | 0.297 | 0.274 |
| gray | 0.070 | 0.065 | 0.222 | 0.292 | 0.114 | 0.173 |
| noise | 0.081 | 0.086 | 0.200 | 0.259 | 0.097 | 0.161 |

**rtdetr**

| texture | none | light | shadow | rain | mud | corr. avg |
|---|---|---|---|---|---|---|
| cpa_none | 0.381 | 0.357 | 0.447 | 0.598 | 0.566 | 0.492 |
| cpa_light | 0.377 | 0.340 | 0.455 | 0.586 | 0.541 | 0.481 |
| cpa_shadow | 0.352 | 0.324 | 0.402 | 0.570 | 0.533 | 0.457 |
| cpa_ls | 0.402 | 0.340 | 0.439 | 0.594 | 0.561 | 0.484 |
| gray | 0.156 | 0.205 | 0.258 | 0.492 | 0.414 | 0.342 |
| noise | 0.164 | 0.176 | 0.270 | 0.459 | 0.385 | 0.323 |

**yolo11s**

| texture | none | light | shadow | rain | mud | corr. avg |
|---|---|---|---|---|---|---|
| cpa_none | 0.463 | 0.441 | 0.590 | 0.729 | 0.628 | 0.597 |
| cpa_light | 0.479 | 0.447 | 0.580 | 0.713 | 0.660 | 0.600 |
| cpa_shadow | 0.468 | 0.484 | 0.564 | 0.718 | 0.622 | 0.597 |
| cpa_ls | 0.495 | 0.484 | 0.580 | 0.713 | 0.660 | 0.609 |
| gray | 0.335 | 0.324 | 0.537 | 0.590 | 0.606 | 0.515 |
| noise | 0.255 | 0.271 | 0.415 | 0.489 | 0.410 | 0.396 |

**yolov8m**

| texture | none | light | shadow | rain | mud | corr. avg |
|---|---|---|---|---|---|---|
| cpa_none | 0.281 | 0.258 | 0.357 | 0.489 | 0.493 | 0.399 |
| cpa_light | 0.258 | 0.258 | 0.362 | 0.466 | 0.480 | 0.391 |
| cpa_shadow | 0.281 | 0.253 | 0.367 | 0.493 | 0.502 | 0.404 |
| cpa_ls | 0.262 | 0.240 | 0.367 | 0.475 | 0.448 | 0.382 |
| gray | 0.190 | 0.181 | 0.290 | 0.389 | 0.376 | 0.309 |
| noise | 0.122 | 0.131 | 0.204 | 0.330 | 0.285 | 0.238 |

### D. Paired tests vs `cpa_none` (difference in ASR; photos sign-flipped; Holm per detector)

| detector | texture | condition | diff [95% CI] | photos that differ | p | Holm p |
|---|---|---|---|---|---|---|
| surrogate | cpa_light | none | +0.016 [+0.00, +0.04] | 2 (min p 0.5) | 0.5 | 1 |
| surrogate | cpa_light | corr. avg | -0.003 [-0.01, +0.01] | 8 (min p 0.00781) | 0.797 | 1 |
| surrogate | cpa_shadow | none | +0.038 [+0.01, +0.07] | 4 (min p 0.125) | 0.125 | 0.75 |
| surrogate | cpa_shadow | corr. avg | +0.008 [-0.01, +0.02] | 9 (min p 0.00391) | 0.5 | 1 |
| surrogate | cpa_ls | none | +0.022 [+0.01, +0.04] | 4 (min p 0.125) | 0.125 | 0.75 |
| surrogate | cpa_ls | corr. avg | +0.008 [-0.01, +0.02] | 11 (min p 0.000977) | 0.501 | 1 |
| rtdetr | cpa_light | none | -0.004 [-0.05, +0.04] | 8 (min p 0.00781) | 1 | 1 |
| rtdetr | cpa_light | corr. avg | -0.011 [-0.04, +0.01] | 13 (min p 0.000244) | 0.352 | 1 |
| rtdetr | cpa_shadow | none | -0.029 [-0.06, -0.01] | 5 (min p 0.0625) | 0.0625 | 0.312 |
| rtdetr | cpa_shadow | corr. avg | -0.035 [-0.06, -0.02] | 13 (min p 0.000244) | 0.00146 | 0.00879 |
| rtdetr | cpa_ls | none | +0.020 [-0.02, +0.07] | 6 (min p 0.0312) | 0.812 | 1 |
| rtdetr | cpa_ls | corr. avg | -0.008 [-0.03, +0.01] | 13 (min p 0.000244) | 0.493 | 1 |
| yolo11s | cpa_light | none | +0.016 [-0.03, +0.06] | 3 (min p 0.25) | 0.75 | 1 |
| yolo11s | cpa_light | corr. avg | +0.003 [-0.02, +0.02] | 8 (min p 0.00781) | 0.898 | 1 |
| yolo11s | cpa_shadow | none | +0.005 [-0.03, +0.04] | 5 (min p 0.0625) | 1 | 1 |
| yolo11s | cpa_shadow | corr. avg | +0.000 [-0.02, +0.02] | 8 (min p 0.00781) | 1 | 1 |
| yolo11s | cpa_ls | none | +0.032 [-0.02, +0.07] | 8 (min p 0.00781) | 0.383 | 1 |
| yolo11s | cpa_ls | corr. avg | +0.012 [-0.03, +0.04] | 10 (min p 0.00195) | 0.568 | 1 |
| yolov8m | cpa_light | none | -0.023 [-0.06, +0.02] | 7 (min p 0.0156) | 0.438 | 1 |
| yolov8m | cpa_light | corr. avg | -0.008 [-0.02, +0.00] | 9 (min p 0.00391) | 0.281 | 1 |
| yolov8m | cpa_shadow | none | +0.000 [-0.03, +0.04] | 5 (min p 0.0625) | 1 | 1 |
| yolov8m | cpa_shadow | corr. avg | +0.005 [-0.01, +0.02] | 10 (min p 0.00195) | 0.598 | 1 |
| yolov8m | cpa_ls | none | -0.018 [-0.07, +0.02] | 5 (min p 0.0625) | 0.688 | 1 |
| yolov8m | cpa_ls | corr. avg | -0.017 [-0.05, +0.00] | 11 (min p 0.000977) | 0.185 | 1 |
