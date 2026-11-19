# NPA naturalness proxies (mean over folds)

| patch | folds | final latent l-inf | saturated pixels % (start -> final) | drift from start |
|---|---|---|---|---|
| npa  class 259, tau 10 (main run) | 5 | 4.33 | 0.9 -> 98.9 | 0.335 |
| npa_c259_t2  class 259, tau 2 | 5 | 2.00 | 1.2 -> 97.2 | 0.305 |
| npa_c259_t4  class 259, tau 4 | 5 | 3.90 | 0.9 -> 98.4 | 0.339 |
| npa_c958_t4  class 958, tau 4 | 5 | 3.60 | 0.0 -> 100.0 | 0.405 |
| npa_c977_t4  class 977, tau 4 | 5 | 4.00 | 0.0 -> 100.0 | 0.422 |
| npa_coarse  class 977, tau 3, z on sphere, chunks 0,1,2 | 5 | 2.81 | 0.0 -> 0.0 | 0.078 |
| npa_satpen  class 977, tau 3, z on sphere, sat-penalty 1 | 5 | 2.78 | 0.0 -> 0.0 | 0.260 |
| npa_sph  class 977, tau 3, z on sphere | 5 | 2.86 | 0.0 -> 87.5 | 0.325 |
| npa_stop  class 977, tau 3, z on sphere, natural-stop 20% | 5 | 2.98 | 0.0 -> 7.8 | 0.155 |
| npa_stop10  class 977, tau 3, z on sphere, natural-stop 10% | 5 | -- | 0.0 -> 1.2 | 0.138 |
| npa_stop100  class 977, tau 3, z on sphere, natural-stop 100% | 5 | -- | 0.0 -> 99.0 | 0.363 |
| npa_stop20  class 977, tau 3, z on sphere, natural-stop 20% | 5 | -- | 0.0 -> 10.4 | 0.164 |
| npa_stop40  class 977, tau 3, z on sphere, natural-stop 40% | 5 | -- | 0.0 -> 24.3 | 0.198 |
| npa_stop5  class 977, tau 3, z on sphere, natural-stop 5% | 5 | -- | 0.0 -> 1.2 | 0.138 |
| camo_auto  camouflage, palette auto, 1 m | 5 | -- | 0.0 -> 0.0 | 0.129 |
| camo_auto_3m  camouflage, palette auto, 3 m | 5 | -- | 0.0 -> 0.0 | 0.095 |
| camo_wood  camouflage, palette woodland, 1 m | 5 | -- | 0.0 -> 0.0 | 0.095 |
| camo_wood_3m  camouflage, palette woodland, 3 m | 5 | -- | 0.0 -> 0.0 | 0.072 |
| pixel (no GAN) | 5 | -- | 17.6 -> 64.8 | 0.376 |
