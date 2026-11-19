# Data

The course dataset (Aerial Image Dataset: Aircraft, Building, Military Truck, Tank; Training / Validation /
Testing, each with `Images/` and `Labels/`) is not redistributed here. Place it as

```
data/Aerial Image Dataset/
    Training data/{Images,Labels}
    Validation data/{Images,Labels}
    Testing data/{Images,Labels}
```

then run `python src/prepare_dataset.py`, which writes `data/Aerial Image Dataset/yolo_dataset/` (YOLO boxes,
`data.yaml`). The 3,168 training files are augmentations of 88 source photos; `src/make_group_split.py` builds
the photo-grouped folds in `yolo_dataset/cv/` used by every experiment.
