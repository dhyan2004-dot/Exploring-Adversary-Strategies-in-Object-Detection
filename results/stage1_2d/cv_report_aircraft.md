# Cross-validated results

## 1. Detector (5-fold, photo-grouped)

**held-out photos** — 5 folds
| fold | mAP50 | mAP50-95 | Aircraft AP50 | Building AP50 | Military Truck AP50 | Tank AP50 |
|---|---|---|---|---|---|---|
| 0 | 0.225 | 0.113 | 0.184 | 0.416 | 0.228 | 0.074 |
| 1 | 0.247 | 0.126 | 0.383 | 0.225 | 0.101 | 0.277 |
| 2 | 0.175 | 0.093 | 0.418 | 0.203 | 0.010 | 0.069 |
| 3 | 0.252 | 0.138 | 0.466 | 0.259 | 0.036 | 0.249 |
| 4 | 0.253 | 0.142 | 0.599 | 0.281 | 0.019 | 0.112 |
| **mean ± sd** | 0.230 ± 0.033 | 0.122 ± 0.020 | 0.410 ± 0.151 | 0.277 ± 0.083 | 0.079 ± 0.091 | 0.156 ± 0.100 |

**official test split** — 5 folds
| fold | mAP50 | mAP50-95 | Aircraft AP50 | Building AP50 | Military Truck AP50 | Tank AP50 |
|---|---|---|---|---|---|---|
| 0 | 0.421 | 0.228 | 0.751 | 0.554 | 0.079 | 0.301 |
| 1 | 0.418 | 0.200 | 0.767 | 0.470 | 0.199 | 0.238 |
| 2 | 0.461 | 0.227 | 0.774 | 0.459 | 0.149 | 0.462 |
| 3 | 0.491 | 0.225 | 0.736 | 0.540 | 0.160 | 0.530 |
| 4 | 0.398 | 0.194 | 0.806 | 0.481 | 0.085 | 0.221 |
| **mean ± sd** | 0.438 ± 0.037 | 0.215 ± 0.016 | 0.767 ± 0.026 | 0.501 ± 0.043 | 0.134 ± 0.051 | 0.350 ± 0.138 |

## 2. Patches (pooled over folds [0, 1, 2, 3, 4])

detected instances: 280, photos: 21, photos per fold: {0: 4, 1: 5, 2: 3, 3: 4, 4: 5}, threshold: 0.25, seeds: {'eot': 1, 'cpa_ls': 1}


### 2a. Attack success vs. the best control (gray / noise square of the same size and place)

| method | condition | patch hides object | best control | **margin** | margin 95% CI (photo-resampled) | photo-weighted margin [95% CI] |
|---|---|---|---|---|---|---|
| eot | none (no corruption) | 0.843 [0.75, 0.92] | 0.782 | **+0.061** | [-0.012, +0.157] | +0.081 [-0.010, +0.191] |
| eot | light+shadow+rain+mud (avg) | 0.787 [0.68, 0.88] | 0.648 | **+0.138** | [+0.048, +0.222] | +0.095 [-0.009, +0.201] |
| eot | mixed (2 stacked) | 0.736 [0.62, 0.84] | 0.568 | **+0.168** | [+0.064, +0.268] | +0.116 [+0.003, +0.220] |
| cpa_ls | none (no corruption) | 0.818 [0.72, 0.90] | 0.782 | **+0.036** | [-0.049, +0.123] | +0.052 [-0.037, +0.148] |
| cpa_ls | light+shadow+rain+mud (avg) | 0.775 [0.67, 0.86] | 0.648 | **+0.127** | [+0.027, +0.211] | +0.084 [-0.005, +0.173] |
| cpa_ls | mixed (2 stacked) | 0.711 [0.58, 0.82] | 0.568 | **+0.143** | [+0.011, +0.254] | +0.111 [+0.005, +0.206] |

### 2b. Every condition (patch hides object / margin over best control)

| method | none | light | shadow | rain | mud | mixed |
|---|---|---|---|---|---|---|
| eot | 0.84 / +0.06 | 0.77 / +0.07 | 0.78 / +0.08 | 0.82 / +0.14 | 0.78 / +0.26 | 0.74 / +0.17 |
| cpa_ls | 0.82 / +0.04 | 0.78 / +0.09* | 0.76 / +0.07* | 0.80 / +0.12 | 0.75 / +0.23 | 0.71 / +0.14 |

`*` corruption used in training

### 2c. Paired comparison between methods (same instances, same corruption draws)


| A | B | condition | A | B | B − A | 95% CI | photos that differ | smallest possible p | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| eot | cpa_ls | none (no corruption) | 0.843 | 0.818 | -0.025 | [-0.068, +0.012] | 6 | 0.031 | 0.3125 | 0.8438 |
| eot | cpa_ls | light+shadow+rain+mud (avg) | 0.787 | 0.775 | -0.012 | [-0.036, +0.013] | 14 | 0.000 | 0.4200 | 0.8438 |
| eot | cpa_ls | mixed (2 stacked) | 0.736 | 0.711 | -0.025 | [-0.066, +0.006] | 9 | 0.004 | 0.2812 | 0.8438 |


### 2d. Consistency across folds (margin over best control, no corruption)

| method | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean ± sd |
|---|---|---|---|---|---|---|
| eot | +0.00 | -0.03 | +0.05 | +0.33 | +0.00 | 0.071 ± 0.149 |
| cpa_ls | +0.00 | -0.03 | +0.05 | +0.24 | -0.05 | 0.044 ± 0.116 |

## 3. Full-image mAP50 drop with the patch on every Aircraft instance (held-out photos)


| method | clean Aircraft AP50 | adversarial: Aircraft AP50 drop | gray | noise | adversarial: overall mAP50 drop | gray | noise |
|---|---|---|---|---|---|---|---|
| cpa_ls | 0.431 ± 0.213 | 0.421 ± 0.210 | 0.378 ± 0.203 | 0.339 ± 0.185 | 0.119 ± 0.056 | 0.096 ± 0.051 | 0.086 ± 0.047 |
| eot | 0.431 ± 0.213 | 0.421 ± 0.212 | 0.378 ± 0.203 | 0.339 ± 0.185 | 0.120 ± 0.056 | 0.096 ± 0.051 | 0.086 ± 0.047 |


