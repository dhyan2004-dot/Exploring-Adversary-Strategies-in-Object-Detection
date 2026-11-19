# 3-D patch attack results (pooled over folds)

Runs: detectors: surrogate (5 folds), rtdetr (5 folds), yolo11s (5 folds), yolov8m (5 folds)


## Threshold 0.5

### A. ASR, no corruption [95% photo-level CI]

| texture | surrogate (n=143, 13 photos) | rtdetr (n=158, 17 photos) | yolo11s (n=135, 14 photos) | yolov8m (n=178, 16 photos) |
|---|---|---|---|---|
| camo_wood_3m | 0.755 [0.51, 0.93] | 0.614 [0.43, 0.84] | 0.807 [0.52, 0.99] | 0.843 [0.68, 0.95] |
| camo_auto_3m | 0.727 [0.52, 0.91] | 0.595 [0.40, 0.79] | 0.807 [0.52, 0.99] | 0.792 [0.63, 0.91] |
| camo_auto_3m_init | 0.343 [0.18, 0.49] | 0.506 [0.30, 0.73] | 0.474 [0.28, 0.69] | 0.652 [0.48, 0.79] |
| camo_wood_3m_init | 0.427 [0.24, 0.57] | 0.576 [0.37, 0.81] | 0.674 [0.41, 0.89] | 0.742 [0.56, 0.87] |
| *gray control* | 0.790 | 0.709 | 0.763 | 0.882 |
| *noise control* | 0.434 | 0.304 | 0.304 | 0.556 |

### B. Margin over the best control, no corruption [95% CI]

| texture | surrogate | rtdetr | yolo11s | yolov8m |
|---|---|---|---|---|
| camo_wood_3m | -0.035 [-0.28, +0.17] | -0.095 [-0.29, +0.07] | +0.044 [-0.03, +0.12] | -0.039 [-0.18, +0.06] |
| camo_auto_3m | -0.063 [-0.25, +0.12] | -0.114 [-0.27, +0.01] | +0.044 [-0.03, +0.12] | -0.090 [-0.22, +0.00] |
| camo_auto_3m_init | -0.448 [-0.59, -0.32] | -0.203 [-0.44, -0.02] | -0.289 [-0.41, -0.14] | -0.230 [-0.38, -0.12] |
| camo_wood_3m_init | -0.364 [-0.52, -0.25] | -0.133 [-0.35, +0.06] | -0.089 [-0.17, -0.01] | -0.140 [-0.29, -0.03] |

### C. ASR per test condition


**surrogate**

| texture | none | light | shadow | rain | mud | corr. avg |
|---|---|---|---|---|---|---|
| camo_wood_3m | 0.755 | 0.685 | 0.776 | 0.860 | 0.818 | 0.785 |
| camo_auto_3m | 0.727 | 0.629 | 0.713 | 0.811 | 0.762 | 0.729 |
| camo_auto_3m_init | 0.343 | 0.371 | 0.364 | 0.587 | 0.406 | 0.432 |
| camo_wood_3m_init | 0.427 | 0.434 | 0.469 | 0.664 | 0.503 | 0.517 |
| gray | 0.790 | 0.720 | 0.783 | 0.811 | 0.818 | 0.783 |
| noise | 0.434 | 0.392 | 0.469 | 0.517 | 0.469 | 0.462 |

**rtdetr**

| texture | none | light | shadow | rain | mud | corr. avg |
|---|---|---|---|---|---|---|
| camo_wood_3m | 0.614 | 0.595 | 0.620 | 0.734 | 0.703 | 0.663 |
| camo_auto_3m | 0.595 | 0.614 | 0.589 | 0.715 | 0.677 | 0.649 |
| camo_auto_3m_init | 0.506 | 0.544 | 0.525 | 0.652 | 0.646 | 0.592 |
| camo_wood_3m_init | 0.576 | 0.544 | 0.570 | 0.658 | 0.671 | 0.611 |
| gray | 0.709 | 0.696 | 0.671 | 0.677 | 0.759 | 0.701 |
| noise | 0.304 | 0.297 | 0.367 | 0.563 | 0.500 | 0.432 |

**yolo11s**

| texture | none | light | shadow | rain | mud | corr. avg |
|---|---|---|---|---|---|---|
| camo_wood_3m | 0.807 | 0.719 | 0.800 | 0.844 | 0.874 | 0.809 |
| camo_auto_3m | 0.807 | 0.733 | 0.800 | 0.837 | 0.837 | 0.802 |
| camo_auto_3m_init | 0.474 | 0.415 | 0.533 | 0.733 | 0.681 | 0.591 |
| camo_wood_3m_init | 0.674 | 0.504 | 0.652 | 0.793 | 0.748 | 0.674 |
| gray | 0.763 | 0.763 | 0.859 | 0.837 | 0.904 | 0.841 |
| noise | 0.304 | 0.311 | 0.385 | 0.630 | 0.519 | 0.461 |

**yolov8m**

| texture | none | light | shadow | rain | mud | corr. avg |
|---|---|---|---|---|---|---|
| camo_wood_3m | 0.843 | 0.730 | 0.831 | 0.860 | 0.888 | 0.827 |
| camo_auto_3m | 0.792 | 0.663 | 0.787 | 0.820 | 0.837 | 0.777 |
| camo_auto_3m_init | 0.652 | 0.461 | 0.618 | 0.680 | 0.736 | 0.624 |
| camo_wood_3m_init | 0.742 | 0.567 | 0.708 | 0.770 | 0.792 | 0.709 |
| gray | 0.882 | 0.848 | 0.904 | 0.899 | 0.938 | 0.897 |
| noise | 0.556 | 0.494 | 0.539 | 0.663 | 0.590 | 0.572 |

### D. Paired tests vs `camo_wood_3m` (difference in ASR; photos sign-flipped; Holm per detector)

| detector | texture | condition | diff [95% CI] | photos that differ | p | Holm p |
|---|---|---|---|---|---|---|
| surrogate | camo_auto_3m | none | -0.028 [-0.16, +0.07] | 6 (min p 0.0312) | 0.906 | 1 |
| surrogate | camo_auto_3m | corr. avg | -0.056 [-0.18, +0.03] | 8 (min p 0.00781) | 0.562 | 1 |
| surrogate | camo_auto_3m_init | none | -0.413 [-0.59, -0.27] | 11 (min p 0.000977) | 0.000977 | 0.00391 |
| surrogate | camo_auto_3m_init | corr. avg | -0.353 [-0.49, -0.22] | 11 (min p 0.000977) | 0.000977 | 0.00391 |
| surrogate | camo_wood_3m_init | none | -0.329 [-0.49, -0.20] | 12 (min p 0.000488) | 0.000488 | 0.00293 |
| surrogate | camo_wood_3m_init | corr. avg | -0.267 [-0.38, -0.15] | 12 (min p 0.000488) | 0.000488 | 0.00293 |
| rtdetr | camo_auto_3m | none | -0.019 [-0.13, +0.06] | 8 (min p 0.00781) | 0.781 | 1 |
| rtdetr | camo_auto_3m | corr. avg | -0.014 [-0.11, +0.08] | 12 (min p 0.000488) | 0.817 | 1 |
| rtdetr | camo_auto_3m_init | none | -0.108 [-0.23, -0.03] | 11 (min p 0.000977) | 0.0244 | 0.122 |
| rtdetr | camo_auto_3m_init | corr. avg | -0.071 [-0.14, -0.02] | 12 (min p 0.000488) | 0.0259 | 0.122 |
| rtdetr | camo_wood_3m_init | none | -0.038 [-0.09, +0.01] | 7 (min p 0.0156) | 0.266 | 0.797 |
| rtdetr | camo_wood_3m_init | corr. avg | -0.052 [-0.09, -0.02] | 11 (min p 0.000977) | 0.00391 | 0.0234 |
| yolo11s | camo_auto_3m | none | +0.000 [+0.00, +0.00] | 0 (min p 1) | 1 | 1 |
| yolo11s | camo_auto_3m | corr. avg | -0.007 [-0.03, +0.01] | 6 (min p 0.0312) | 0.531 | 1 |
| yolo11s | camo_auto_3m_init | none | -0.333 [-0.50, -0.13] | 6 (min p 0.0312) | 0.0312 | 0.125 |
| yolo11s | camo_auto_3m_init | corr. avg | -0.219 [-0.33, -0.12] | 12 (min p 0.000488) | 0.000488 | 0.00293 |
| yolo11s | camo_wood_3m_init | none | -0.133 [-0.26, -0.03] | 4 (min p 0.125) | 0.125 | 0.375 |
| yolo11s | camo_wood_3m_init | corr. avg | -0.135 [-0.22, -0.07] | 10 (min p 0.00195) | 0.00195 | 0.00977 |
| yolov8m | camo_auto_3m | none | -0.051 [-0.12, +0.01] | 8 (min p 0.00781) | 0.203 | 0.254 |
| yolov8m | camo_auto_3m | corr. avg | -0.051 [-0.12, +0.00] | 10 (min p 0.00195) | 0.127 | 0.254 |
| yolov8m | camo_auto_3m_init | none | -0.191 [-0.30, -0.08] | 8 (min p 0.00781) | 0.0234 | 0.0703 |
| yolov8m | camo_auto_3m_init | corr. avg | -0.204 [-0.29, -0.12] | 13 (min p 0.000244) | 0.000244 | 0.00146 |
| yolov8m | camo_wood_3m_init | none | -0.101 [-0.18, -0.05] | 7 (min p 0.0156) | 0.0156 | 0.0625 |
| yolov8m | camo_wood_3m_init | corr. avg | -0.118 [-0.18, -0.07] | 12 (min p 0.000488) | 0.00195 | 0.00977 |

## Threshold 0.25

### A. ASR, no corruption [95% photo-level CI]

| texture | surrogate (n=185, 17 photos) | rtdetr (n=244, 21 photos) | yolo11s (n=188, 18 photos) | yolov8m (n=221, 19 photos) |
|---|---|---|---|---|
| camo_wood_3m | 0.670 [0.46, 0.85] | 0.541 [0.30, 0.78] | 0.824 [0.57, 0.98] | 0.733 [0.54, 0.88] |
| camo_auto_3m | 0.605 [0.42, 0.76] | 0.504 [0.26, 0.74] | 0.819 [0.57, 0.97] | 0.715 [0.51, 0.87] |
| camo_auto_3m_init | 0.211 [0.11, 0.28] | 0.430 [0.22, 0.64] | 0.436 [0.28, 0.62] | 0.489 [0.34, 0.65] |
| camo_wood_3m_init | 0.276 [0.15, 0.38] | 0.467 [0.25, 0.70] | 0.606 [0.39, 0.80] | 0.611 [0.44, 0.77] |
| *gray control* | 0.622 | 0.631 | 0.771 | 0.765 |
| *noise control* | 0.205 | 0.221 | 0.261 | 0.412 |

### B. Margin over the best control, no corruption [95% CI]

| texture | surrogate | rtdetr | yolo11s | yolov8m |
|---|---|---|---|---|
| camo_wood_3m | +0.049 [-0.12, +0.26] | -0.090 [-0.26, +0.06] | +0.053 [+0.01, +0.11] | -0.032 [-0.15, +0.06] |
| camo_auto_3m | -0.016 [-0.13, +0.14] | -0.127 [-0.29, +0.01] | +0.048 [-0.01, +0.10] | -0.050 [-0.17, +0.04] |
| camo_auto_3m_init | -0.411 [-0.47, -0.32] | -0.201 [-0.36, -0.07] | -0.335 [-0.46, -0.17] | -0.276 [-0.41, -0.14] |
| camo_wood_3m_init | -0.346 [-0.41, -0.25] | -0.164 [-0.33, -0.02] | -0.165 [-0.27, -0.06] | -0.154 [-0.27, -0.04] |

### C. ASR per test condition


**surrogate**

| texture | none | light | shadow | rain | mud | corr. avg |
|---|---|---|---|---|---|---|
| camo_wood_3m | 0.670 | 0.541 | 0.681 | 0.778 | 0.681 | 0.670 |
| camo_auto_3m | 0.605 | 0.497 | 0.616 | 0.595 | 0.654 | 0.591 |
| camo_auto_3m_init | 0.211 | 0.227 | 0.297 | 0.432 | 0.265 | 0.305 |
| camo_wood_3m_init | 0.276 | 0.281 | 0.314 | 0.503 | 0.335 | 0.358 |
| gray | 0.622 | 0.611 | 0.686 | 0.681 | 0.676 | 0.664 |
| noise | 0.205 | 0.195 | 0.259 | 0.303 | 0.249 | 0.251 |

**rtdetr**

| texture | none | light | shadow | rain | mud | corr. avg |
|---|---|---|---|---|---|---|
| camo_wood_3m | 0.541 | 0.443 | 0.529 | 0.676 | 0.611 | 0.565 |
| camo_auto_3m | 0.504 | 0.471 | 0.500 | 0.684 | 0.623 | 0.570 |
| camo_auto_3m_init | 0.430 | 0.398 | 0.430 | 0.594 | 0.553 | 0.494 |
| camo_wood_3m_init | 0.467 | 0.406 | 0.480 | 0.635 | 0.586 | 0.527 |
| gray | 0.631 | 0.602 | 0.664 | 0.676 | 0.709 | 0.663 |
| noise | 0.221 | 0.172 | 0.299 | 0.500 | 0.361 | 0.333 |

**yolo11s**

| texture | none | light | shadow | rain | mud | corr. avg |
|---|---|---|---|---|---|---|
| camo_wood_3m | 0.824 | 0.723 | 0.809 | 0.872 | 0.862 | 0.816 |
| camo_auto_3m | 0.819 | 0.702 | 0.803 | 0.851 | 0.867 | 0.806 |
| camo_auto_3m_init | 0.436 | 0.367 | 0.543 | 0.665 | 0.644 | 0.555 |
| camo_wood_3m_init | 0.606 | 0.516 | 0.654 | 0.761 | 0.734 | 0.666 |
| gray | 0.771 | 0.734 | 0.824 | 0.840 | 0.915 | 0.828 |
| noise | 0.261 | 0.266 | 0.330 | 0.473 | 0.383 | 0.363 |

**yolov8m**

| texture | none | light | shadow | rain | mud | corr. avg |
|---|---|---|---|---|---|---|
| camo_wood_3m | 0.733 | 0.611 | 0.733 | 0.801 | 0.792 | 0.734 |
| camo_auto_3m | 0.715 | 0.575 | 0.674 | 0.751 | 0.738 | 0.684 |
| camo_auto_3m_init | 0.489 | 0.290 | 0.443 | 0.638 | 0.588 | 0.490 |
| camo_wood_3m_init | 0.611 | 0.421 | 0.579 | 0.683 | 0.643 | 0.581 |
| gray | 0.765 | 0.765 | 0.783 | 0.842 | 0.878 | 0.817 |
| noise | 0.412 | 0.321 | 0.407 | 0.516 | 0.448 | 0.423 |

### D. Paired tests vs `camo_wood_3m` (difference in ASR; photos sign-flipped; Holm per detector)

| detector | texture | condition | diff [95% CI] | photos that differ | p | Holm p |
|---|---|---|---|---|---|---|
| surrogate | camo_auto_3m | none | -0.065 [-0.19, +0.04] | 9 (min p 0.00391) | 0.441 | 0.441 |
| surrogate | camo_auto_3m | corr. avg | -0.080 [-0.19, +0.01] | 13 (min p 0.000244) | 0.16 | 0.319 |
| surrogate | camo_auto_3m_init | none | -0.459 [-0.62, -0.30] | 14 (min p 0.000122) | 0.000122 | 0.00061 |
| surrogate | camo_auto_3m_init | corr. avg | -0.365 [-0.48, -0.25] | 15 (min p 6.1e-05) | 6.1e-05 | 0.000366 |
| surrogate | camo_wood_3m_init | none | -0.395 [-0.55, -0.26] | 13 (min p 0.000244) | 0.000244 | 0.000732 |
| surrogate | camo_wood_3m_init | corr. avg | -0.312 [-0.44, -0.20] | 14 (min p 0.000122) | 0.000122 | 0.00061 |
| rtdetr | camo_auto_3m | none | -0.037 [-0.13, +0.03] | 12 (min p 0.000488) | 0.417 | 0.834 |
| rtdetr | camo_auto_3m | corr. avg | +0.005 [-0.07, +0.07] | 16 (min p 3.05e-05) | 0.928 | 0.928 |
| rtdetr | camo_auto_3m_init | none | -0.111 [-0.19, -0.03] | 15 (min p 6.1e-05) | 0.0292 | 0.117 |
| rtdetr | camo_auto_3m_init | corr. avg | -0.071 [-0.12, -0.02] | 18 (min p 7.63e-06) | 0.0224 | 0.112 |
| rtdetr | camo_wood_3m_init | none | -0.074 [-0.13, -0.03] | 9 (min p 0.00391) | 0.0156 | 0.0938 |
| rtdetr | camo_wood_3m_init | corr. avg | -0.038 [-0.07, -0.01] | 16 (min p 3.05e-05) | 0.0327 | 0.117 |
| yolo11s | camo_auto_3m | none | -0.005 [-0.03, +0.02] | 5 (min p 0.0625) | 1 | 1 |
| yolo11s | camo_auto_3m | corr. avg | -0.011 [-0.03, +0.00] | 8 (min p 0.00781) | 0.211 | 0.422 |
| yolo11s | camo_auto_3m_init | none | -0.388 [-0.53, -0.19] | 8 (min p 0.00781) | 0.00781 | 0.0312 |
| yolo11s | camo_auto_3m_init | corr. avg | -0.262 [-0.37, -0.15] | 13 (min p 0.000244) | 0.000244 | 0.00146 |
| yolo11s | camo_wood_3m_init | none | -0.218 [-0.34, -0.08] | 7 (min p 0.0156) | 0.0156 | 0.0469 |
| yolo11s | camo_wood_3m_init | corr. avg | -0.150 [-0.25, -0.07] | 10 (min p 0.00195) | 0.00195 | 0.00977 |
| yolov8m | camo_auto_3m | none | -0.018 [-0.07, +0.02] | 10 (min p 0.00195) | 0.533 | 0.533 |
| yolov8m | camo_auto_3m | corr. avg | -0.050 [-0.09, -0.01] | 11 (min p 0.000977) | 0.0723 | 0.145 |
| yolov8m | camo_auto_3m_init | none | -0.244 [-0.38, -0.10] | 11 (min p 0.000977) | 0.00293 | 0.0146 |
| yolov8m | camo_auto_3m_init | corr. avg | -0.244 [-0.34, -0.15] | 16 (min p 3.05e-05) | 0.000153 | 0.000916 |
| yolov8m | camo_wood_3m_init | none | -0.122 [-0.19, -0.05] | 7 (min p 0.0156) | 0.0156 | 0.0469 |
| yolov8m | camo_wood_3m_init | corr. avg | -0.153 [-0.22, -0.08] | 17 (min p 1.53e-05) | 0.00375 | 0.015 |
