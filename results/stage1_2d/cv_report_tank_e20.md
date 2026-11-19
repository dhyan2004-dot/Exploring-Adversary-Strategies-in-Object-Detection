# Cross-validated results, 20 patch epochs

## 1. Detector (5-fold, photo-grouped)

**held-out photos** — 5 folds
| fold | mAP50 | mAP50-95 | Aircraft AP50 | Building AP50 | Military Truck AP50 | Tank AP50 |
|---|---|---|---|---|---|---|
| 0 | 0.186 | 0.095 | 0.144 | 0.377 | 0.162 | 0.060 |
| 1 | 0.259 | 0.146 | 0.241 | 0.285 | 0.077 | 0.433 |
| 2 | 0.189 | 0.104 | 0.304 | 0.127 | 0.001 | 0.325 |
| 3 | 0.163 | 0.085 | 0.312 | 0.252 | 0.053 | 0.033 |
| 4 | 0.177 | 0.105 | 0.442 | 0.204 | 0.010 | 0.052 |
| **mean ± sd** | 0.195 ± 0.037 | 0.107 ± 0.023 | 0.288 ± 0.109 | 0.249 ± 0.093 | 0.060 ± 0.065 | 0.180 ± 0.185 |

**official test split** — 5 folds
| fold | mAP50 | mAP50-95 | Aircraft AP50 | Building AP50 | Military Truck AP50 | Tank AP50 |
|---|---|---|---|---|---|---|
| 0 | 0.473 | 0.278 | 0.633 | 0.642 | 0.081 | 0.537 |
| 1 | 0.421 | 0.229 | 0.622 | 0.393 | 0.069 | 0.599 |
| 2 | 0.440 | 0.237 | 0.592 | 0.635 | 0.076 | 0.458 |
| 3 | 0.447 | 0.249 | 0.622 | 0.504 | 0.052 | 0.611 |
| 4 | 0.399 | 0.230 | 0.738 | 0.489 | 0.085 | 0.284 |
| **mean ± sd** | 0.436 ± 0.028 | 0.245 ± 0.020 | 0.641 ± 0.056 | 0.532 ± 0.106 | 0.073 ± 0.013 | 0.498 ± 0.134 |

## 2. Patches (pooled over folds [0, 1, 2, 3, 4])

detected instances: 185, photos: 17, photos per fold: {0: 2, 1: 4, 2: 5, 3: 2, 4: 4}, threshold: 0.25, seeds: {'eot': 1, 'cpa_light': 1, 'cpa_shadow': 1, 'cpa_ls': 1}

### 2a. Attack success vs. the best control (gray / noise square of the same size and place)

| method | condition | patch hides object | best control | **margin** | margin 95% CI  | photo-weighted margin [95% CI] |
|---|---|---|---|---|---|---|
| eot | none (no corruption) | 0.768 [0.58, 0.91] | 0.497 | **+0.270** | [+0.132, +0.380] | +0.330 [+0.178, +0.484] |
| eot | light+shadow+rain+mud (avg) | 0.632 [0.49, 0.77] | 0.445 | **+0.188** | [+0.091, +0.303] | +0.243 [+0.139, +0.352] |
| eot | mixed (2 stacked) | 0.454 [0.36, 0.58] | 0.335 | **+0.119** | [+0.023, +0.244] | +0.272 [+0.105, +0.399] |
| cpa_light | none (no corruption) | 0.724 [0.51, 0.90] | 0.497 | **+0.227** | [+0.063, +0.372] | +0.287 [+0.133, +0.448] |
| cpa_light | light+shadow+rain+mud (avg) | 0.659 [0.48, 0.82] | 0.445 | **+0.215** | [+0.120, +0.312] | +0.249 [+0.150, +0.357] |
| cpa_light | mixed (2 stacked) | 0.524 [0.35, 0.71] | 0.335 | **+0.189** | [+0.098, +0.304] | +0.308 [+0.142, +0.448] |
| cpa_shadow | none (no corruption) | 0.827 [0.70, 0.93] | 0.497 | **+0.330** | [+0.232, +0.424] | +0.343 [+0.202, +0.490] |
| cpa_shadow | light+shadow+rain+mud (avg) | 0.742 [0.60, 0.86] | 0.445 | **+0.297** | [+0.191, +0.392] | +0.279 [+0.179, +0.383] |
| cpa_shadow | mixed (2 stacked) | 0.605 [0.48, 0.73] | 0.335 | **+0.270** | [+0.163, +0.381] | +0.329 [+0.175, +0.447] |
| cpa_ls | none (no corruption) | 0.832 [0.67, 0.95] | 0.497 | **+0.335** | [+0.204, +0.451] | +0.368 [+0.218, +0.523] |
| cpa_ls | light+shadow+rain+mud (avg) | 0.735 [0.56, 0.88] | 0.445 | **+0.291** | [+0.167, +0.409] | +0.312 [+0.201, +0.422] |
| cpa_ls | mixed (2 stacked) | 0.568 [0.42, 0.70] | 0.335 | **+0.232** | [+0.120, +0.360] | +0.337 [+0.170, +0.475] |

### 2b. Every condition (patch hides object / margin over best control)

| method | none | light | shadow | rain | mud | mixed |
|---|---|---|---|---|---|---|
| eot | 0.77 / +0.27 | 0.71 / +0.16 | 0.74 / +0.17 | 0.70 / +0.31 | 0.38 / +0.11 | 0.45 / +0.12 |
| cpa_light | 0.72 / +0.23 | 0.74 / +0.19* | 0.72 / +0.15 | 0.71 / +0.32 | 0.47 / +0.19 | 0.52 / +0.19 |
| cpa_shadow | 0.83 / +0.33 | 0.83 / +0.28 | 0.81 / +0.24* | 0.81 / +0.43 | 0.51 / +0.24 | 0.61 / +0.27 |
| cpa_ls | 0.83 / +0.34 | 0.84 / +0.29* | 0.79 / +0.23* | 0.79 / +0.41 | 0.52 / +0.24 | 0.57 / +0.23 |

`*` corruption used in training

### 2c. Paired comparison between methods (same instances, same corruption draws)

| A | B | condition | A | B | B − A | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| eot | cpa_light | none (no corruption) | 0.768 | 0.724 | -0.043 | [-0.146, +0.031] | 0.6250 | 1.0000 |
| eot | cpa_light | light+shadow+rain+mud (avg) | 0.632 | 0.659 | +0.027 | [-0.051, +0.097] | 0.5645 | 1.0000 |
| eot | cpa_light | mixed (2 stacked) | 0.454 | 0.524 | +0.070 | [-0.068, +0.201] | 0.4688 | 1.0000 |
| eot | cpa_shadow | none (no corruption) | 0.768 | 0.827 | +0.059 | [-0.022, +0.158] | 0.3750 | 1.0000 |
| eot | cpa_shadow | light+shadow+rain+mud (avg) | 0.632 | 0.742 | +0.109 | [+0.029, +0.184] | 0.0552 | 0.7725 |
| eot | cpa_shadow | mixed (2 stacked) | 0.454 | 0.605 | +0.151 | [+0.051, +0.245] | 0.0430 | 0.6445 |
| eot | cpa_ls | none (no corruption) | 0.768 | 0.832 | +0.065 | [+0.012, +0.142] | 0.1250 | 1.0000 |
| eot | cpa_ls | light+shadow+rain+mud (avg) | 0.632 | 0.735 | +0.103 | [+0.024, +0.188] | 0.0625 | 0.7871 |
| eot | cpa_ls | mixed (2 stacked) | 0.454 | 0.568 | +0.114 | [+0.009, +0.221] | 0.1250 | 1.0000 |
| cpa_light | cpa_ls | none (no corruption) | 0.724 | 0.832 | +0.108 | [+0.018, +0.232] | 0.1250 | 1.0000 |
| cpa_light | cpa_ls | light+shadow+rain+mud (avg) | 0.659 | 0.735 | +0.076 | [+0.017, +0.143] | 0.0605 | 0.7871 |
| cpa_light | cpa_ls | mixed (2 stacked) | 0.524 | 0.568 | +0.043 | [-0.037, +0.117] | 0.4219 | 1.0000 |
| cpa_shadow | cpa_ls | none (no corruption) | 0.827 | 0.832 | +0.005 | [-0.051, +0.060] | 1.0000 | 1.0000 |
| cpa_shadow | cpa_ls | light+shadow+rain+mud (avg) | 0.742 | 0.735 | -0.007 | [-0.062, +0.050] | 0.8750 | 1.0000 |
| cpa_shadow | cpa_ls | mixed (2 stacked) | 0.605 | 0.568 | -0.038 | [-0.071, +0.000] | 0.1797 | 1.0000 |


### 2d. Consistency across folds (margin over best control, no corruption)

| method | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean ± sd |
|---|---|---|---|---|---|---|
| eot | +0.44 | +0.20 | +0.25 | +0.67 | +0.32 | 0.376 ± 0.186 |
| cpa_light | +0.44 | +0.21 | +0.08 | +0.67 | +0.32 | 0.343 ± 0.225 |
| cpa_shadow | +0.44 | +0.28 | +0.35 | +0.67 | +0.26 | 0.402 ± 0.164 |
| cpa_ls | +0.44 | +0.24 | +0.43 | +0.67 | +0.32 | 0.419 ± 0.163 |

## 3. Full-image mAP50 drop with the patch on every Tank instance (held-out photos)

| method | clean Tank AP50 | adversarial: Tank AP50 drop | gray | noise | adversarial: overall mAP50 drop | gray | noise |
|---|---|---|---|---|---|---|---|
| cpa_light | 0.156 ± 0.154 | 0.130 ± 0.131 | 0.104 ± 0.097 | 0.089 ± 0.087 | 0.036 ± 0.032 | 0.027 ± 0.025 | 0.022 ± 0.022 |
| cpa_ls | 0.156 ± 0.154 | 0.138 ± 0.143 | 0.104 ± 0.097 | 0.089 ± 0.087 | 0.038 ± 0.036 | 0.027 ± 0.025 | 0.022 ± 0.022 |
| cpa_shadow | 0.156 ± 0.154 | 0.143 ± 0.141 | 0.104 ± 0.097 | 0.089 ± 0.087 | 0.044 ± 0.030 | 0.027 ± 0.025 | 0.022 ± 0.022 |
| eot | 0.156 ± 0.154 | 0.133 ± 0.129 | 0.104 ± 0.097 | 0.089 ± 0.087 | 0.036 ± 0.032 | 0.027 ± 0.025 | 0.022 ± 0.022 |
