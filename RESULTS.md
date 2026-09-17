# Baseline results

The supplied run is a clean YOLOv8s object-detection baseline for the AE5510 UAS aerial-image project.

| Metric | Best logged validation value | Epoch |
|---|---:|---:|
| Precision | 0.90087 | 27 |
| Recall | 0.52270 | 8 |
| mAP@0.50 | 0.55475 | 11 |
| mAP@0.50:0.95 | 0.35007 | 11 |

The project dataset contains four object categories: Aircraft, Military Truck, Tank, and Building. The course project specification calls for a clean detector baseline followed by evaluation of a physical-domain adversarial attack under changes such as viewing angle, scale, illumination, and distance.
