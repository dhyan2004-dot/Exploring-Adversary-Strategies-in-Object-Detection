# Cross-validated results (photo = unit of analysis)

## 1. Detector (5-fold, photo-grouped)

**held-out photos (never trained on; also used to pick the checkpoint -> slightly optimistic)** — 5 folds
| fold | mAP50 | mAP50-95 | Aircraft AP50 | Building AP50 | Military Truck AP50 | Tank AP50 |
|---|---|---|---|---|---|---|
| 0 | 0.186 | 0.095 | 0.144 | 0.377 | 0.162 | 0.060 |
| 1 | 0.259 | 0.146 | 0.241 | 0.285 | 0.077 | 0.433 |
| 2 | 0.189 | 0.104 | 0.304 | 0.127 | 0.001 | 0.325 |
| 3 | 0.163 | 0.085 | 0.312 | 0.252 | 0.053 | 0.033 |
| 4 | 0.177 | 0.105 | 0.442 | 0.204 | 0.010 | 0.052 |
| **mean ± sd** | 0.195 ± 0.037 | 0.107 ± 0.023 | 0.288 ± 0.109 | 0.249 ± 0.093 | 0.060 ± 0.065 | 0.180 ± 0.185 |

**official Testing split (independent of all folds)** — 5 folds
| fold | mAP50 | mAP50-95 | Aircraft AP50 | Building AP50 | Military Truck AP50 | Tank AP50 |
|---|---|---|---|---|---|---|
| 0 | 0.473 | 0.278 | 0.633 | 0.642 | 0.081 | 0.537 |
| 1 | 0.421 | 0.229 | 0.622 | 0.393 | 0.069 | 0.599 |
| 2 | 0.440 | 0.237 | 0.592 | 0.635 | 0.076 | 0.458 |
| 3 | 0.447 | 0.249 | 0.622 | 0.504 | 0.052 | 0.611 |
| 4 | 0.399 | 0.230 | 0.738 | 0.489 | 0.085 | 0.284 |
| **mean ± sd** | 0.436 ± 0.028 | 0.245 ± 0.020 | 0.641 ± 0.056 | 0.532 ± 0.106 | 0.073 ± 0.013 | 0.498 ± 0.134 |

## 2. Patches (pooled over folds [0, 1, 2, 3, 4])

Target instances detected before patching: **185** from **17 distinct photos** (per fold: {0: 2, 1: 4, 2: 5, 3: 2, 4: 4}). Attack success = a detected object drops below conf 0.25. Seeds per method: {'eot': 3, 'cpa_light': 3, 'cpa_shadow': 3, 'cpa_ls': 3}. All intervals resample whole photos.

Headline columns weight every instance equally. The last column weights every PHOTO equally; if the two disagree, a few crowded photos are driving the result.

### 2a. Attack success vs. the best control (gray / noise square of the same size and place)

| method | condition | patch hides object | best control | **margin** | margin 95% CI (photo-resampled) | photo-weighted margin [95% CI] |
|---|---|---|---|---|---|---|
| eot | none (no corruption) | 0.816 [0.67, 0.94] | 0.497 | **+0.319** | [+0.214, +0.414] | +0.345 [+0.206, +0.494] |
| eot | light+shadow+rain+mud (avg) | 0.725 [0.56, 0.88] | 0.445 | **+0.280** | [+0.158, +0.400] | +0.287 [+0.185, +0.390] |
| eot | mixed (2 stacked) | 0.604 [0.44, 0.75] | 0.335 | **+0.268** | [+0.129, +0.417] | +0.333 [+0.173, +0.456] |
| cpa_light | none (no corruption) | 0.795 [0.63, 0.93] | 0.497 | **+0.297** | [+0.173, +0.413] | +0.334 [+0.192, +0.486] |
| cpa_light | light+shadow+rain+mud (avg) | 0.733 [0.55, 0.90] | 0.445 | **+0.289** | [+0.162, +0.413] | +0.298 [+0.193, +0.406] |
| cpa_light | mixed (2 stacked) | 0.625 [0.43, 0.80] | 0.335 | **+0.290** | [+0.163, +0.428] | +0.356 [+0.201, +0.478] |
| cpa_shadow | none (no corruption) | 0.863 [0.74, 0.95] | 0.497 | **+0.366** | [+0.258, +0.454] | +0.380 [+0.238, +0.528] |
| cpa_shadow | light+shadow+rain+mud (avg) | 0.758 [0.60, 0.89] | 0.445 | **+0.314** | [+0.201, +0.423] | +0.324 [+0.220, +0.426] |
| cpa_shadow | mixed (2 stacked) | 0.649 [0.50, 0.80] | 0.335 | **+0.314** | [+0.198, +0.448] | +0.416 [+0.255, +0.540] |
| cpa_ls | none (no corruption) | 0.876 [0.76, 0.96] | 0.497 | **+0.378** | [+0.276, +0.468] | +0.387 [+0.243, +0.535] |
| cpa_ls | light+shadow+rain+mud (avg) | 0.785 [0.64, 0.91] | 0.445 | **+0.340** | [+0.228, +0.445] | +0.334 [+0.228, +0.437] |
| cpa_ls | mixed (2 stacked) | 0.670 [0.50, 0.83] | 0.335 | **+0.335** | [+0.218, +0.468] | +0.428 [+0.266, +0.555] |

### 2b. Every condition (patch hides object / margin over best control)

| method | none | light | shadow | rain | mud | mixed |
|---|---|---|---|---|---|---|
| eot | 0.82 / +0.32 | 0.79 / +0.24 | 0.78 / +0.21 | 0.77 / +0.39 | 0.55 / +0.28 | 0.60 / +0.27 |
| cpa_light | 0.79 / +0.30 | 0.80 / +0.25* | 0.79 / +0.22 | 0.76 / +0.37 | 0.59 / +0.32 | 0.63 / +0.29 |
| cpa_shadow | 0.86 / +0.37 | 0.83 / +0.27 | 0.83 / +0.26* | 0.80 / +0.41 | 0.58 / +0.30 | 0.65 / +0.31 |
| cpa_ls | 0.88 / +0.38 | 0.85 / +0.29* | 0.85 / +0.29* | 0.82 / +0.43 | 0.62 / +0.35 | 0.67 / +0.34 |

`*` = this corruption was applied to the patch during that method's training; unmarked corruption columns were unseen.

### 2c. Paired comparison between methods (same instances, same corruption draws)

Positive difference = the second method hides MORE objects. p = photo-level sign-flip test, Holm-adjusted across this table.

| A | B | condition | A | B | B − A | 95% CI | photos that differ | smallest possible p | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| eot | cpa_light | none (no corruption) | 0.816 | 0.795 | -0.022 | [-0.076, +0.032] | 6 | 0.031 | 0.5312 | 1.0000 |
| eot | cpa_light | light+shadow+rain+mud (avg) | 0.725 | 0.733 | +0.009 | [-0.021, +0.036] | 12 | 0.000 | 0.5947 | 1.0000 |
| eot | cpa_light | mixed (2 stacked) | 0.604 | 0.625 | +0.022 | [-0.034, +0.084] | 9 | 0.004 | 0.6172 | 1.0000 |
| eot | cpa_shadow | none (no corruption) | 0.816 | 0.863 | +0.047 | [-0.000, +0.104] | 7 | 0.016 | 0.1562 | 1.0000 |
| eot | cpa_shadow | light+shadow+rain+mud (avg) | 0.725 | 0.758 | +0.033 | [-0.003, +0.080] | 12 | 0.000 | 0.1406 | 1.0000 |
| eot | cpa_shadow | mixed (2 stacked) | 0.604 | 0.649 | +0.045 | [+0.007, +0.098] | 11 | 0.001 | 0.0479 | 0.5742 |
| eot | cpa_ls | none (no corruption) | 0.816 | 0.876 | +0.059 | [+0.000, +0.135] | 7 | 0.016 | 0.1406 | 1.0000 |
| eot | cpa_ls | light+shadow+rain+mud (avg) | 0.725 | 0.785 | +0.060 | [+0.025, +0.099] | 12 | 0.000 | 0.0151 | 0.2119 |
| eot | cpa_ls | mixed (2 stacked) | 0.604 | 0.670 | +0.067 | [+0.026, +0.119] | 12 | 0.000 | 0.0132 | 0.1978 |
| cpa_light | cpa_ls | none (no corruption) | 0.795 | 0.876 | +0.081 | [+0.008, +0.167] | 6 | 0.031 | 0.1250 | 1.0000 |
| cpa_light | cpa_ls | light+shadow+rain+mud (avg) | 0.733 | 0.785 | +0.051 | [+0.006, +0.101] | 10 | 0.002 | 0.0664 | 0.7305 |
| cpa_light | cpa_ls | mixed (2 stacked) | 0.625 | 0.670 | +0.045 | [+0.013, +0.086] | 11 | 0.001 | 0.0273 | 0.3555 |
| cpa_shadow | cpa_ls | none (no corruption) | 0.863 | 0.876 | +0.013 | [-0.004, +0.038] | 4 | 0.125 | 0.5000 | 1.0000 |
| cpa_shadow | cpa_ls | light+shadow+rain+mud (avg) | 0.758 | 0.785 | +0.027 | [-0.007, +0.055] | 11 | 0.001 | 0.2295 | 1.0000 |
| cpa_shadow | cpa_ls | mixed (2 stacked) | 0.649 | 0.670 | +0.022 | [-0.007, +0.048] | 8 | 0.008 | 0.3125 | 1.0000 |

Only photos where the two methods differ carry information, so the p-value cannot go below 'smallest possible p'. With few photos the bootstrap CI is too narrow; when CI and p disagree, quote the (Holm-adjusted) p.

### 2d. Consistency across folds (margin over best control, no corruption)

| method | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean ± sd |
|---|---|---|---|---|---|---|
| eot | +0.44 | +0.30 | +0.25 | +0.67 | +0.35 | 0.403 ± 0.163 |
| cpa_light | +0.48 | +0.27 | +0.21 | +0.67 | +0.35 | 0.395 ± 0.183 |
| cpa_shadow | +0.48 | +0.29 | +0.42 | +0.67 | +0.33 | 0.439 ± 0.147 |
| cpa_ls | +0.48 | +0.31 | +0.45 | +0.67 | +0.32 | 0.444 ± 0.147 |

### 2e. Seed variance (margin over best control, pooled over all photos; one value per patch seed)

| method | none: per seed | none: mean ± sd | corrupted avg: mean ± sd |
|---|---|---|---|
| eot | +0.35, +0.31, +0.30 | 0.319 ± 0.025 | 0.280 ± 0.024 |
| cpa_light | +0.29, +0.32, +0.28 | 0.297 ± 0.019 | 0.289 ± 0.018 |
| cpa_shadow | +0.42, +0.30, +0.37 | 0.366 ± 0.060 | 0.314 ± 0.056 |
| cpa_ls | +0.37, +0.37, +0.39 | 0.378 ± 0.009 | 0.340 ± 0.029 |

## 3. Full-image mAP50 drop with the patch on every Tank instance (held-out photos)

Mean over folds ± sd (seeds averaged inside a fold). The adversarial drop only counts beyond the gray / noise drops.

| method | clean Tank AP50 | adversarial: Tank AP50 drop | gray | noise | adversarial: overall mAP50 drop | gray | noise |
|---|---|---|---|---|---|---|---|
| cpa_light | 0.156 ± 0.154 | 0.137 ± 0.139 | 0.104 ± 0.097 | 0.089 ± 0.087 | 0.041 ± 0.038 | 0.027 ± 0.025 | 0.022 ± 0.022 |
| cpa_ls | 0.156 ± 0.154 | 0.146 ± 0.144 | 0.104 ± 0.097 | 0.089 ± 0.087 | 0.043 ± 0.034 | 0.027 ± 0.025 | 0.022 ± 0.022 |
| cpa_shadow | 0.156 ± 0.154 | 0.142 ± 0.139 | 0.104 ± 0.097 | 0.089 ± 0.087 | 0.043 ± 0.032 | 0.027 ± 0.025 | 0.022 ± 0.022 |
| eot | 0.156 ± 0.154 | 0.137 ± 0.133 | 0.104 ± 0.097 | 0.089 ± 0.087 | 0.040 ± 0.034 | 0.027 ± 0.025 | 0.022 ± 0.022 |


