"""
prepare_dataset.py
-------------------
Step 1 of the pipeline: turn the raw "Aerial Image Dataset" folders (as
described in Readme.docx) into a clean, YOLOv8-ready dataset.

WHAT THIS SCRIPT DOES
1. Auto-discovers the Training / Validation / Testing sub-folders under
   DATASET_ROOT (it matches on "train" / "valid" / "test" in the folder
   name, so it works whether they're called "Training data",
   "Training Data", "Train", etc.).
2. Inside each split it auto-discovers the "Images" and "Labels" folders.
3. Validates every image <-> label pairing (reports missing/orphan files).
4. Parses every label file. The raw data mixes two annotation formats:
       - standard YOLO box:      class xc yc w h                (5 fields)
       - a 4-point polygon/OBB:  class x1 y1 x2 y2 x3 y3 x4 y4   (9 fields)
   YOLOv8 detection needs the first format only, so any polygon line is
   automatically converted to its axis-aligned bounding box
   (min/max of the x's and y's). This was found by inspecting the
   supplied Validation Data (see e.g. truck-4-..._jpg.rf....txt).
5. Clips any slightly out-of-range (<0 or >1) coordinates that come from
   floating point noise in the source labels.
6. Cross-checks the class-id -> object-name mapping by majority vote
   against the filename prefixes (aircraft-/building-/tank-/truck-) and
   uses the verified mapping in data.yaml, instead of blindly trusting
   the order objects happen to be listed in the Readme.
7. Writes a clean copy of the dataset to
       <DATASET_ROOT>/yolo_dataset/images/{train,val,test}
       <DATASET_ROOT>/yolo_dataset/labels/{train,val,test}
   using lower-case "images"/"labels" folder names (Ultralytics matches
   label paths by literally replacing "images" with "labels" in the
   path, which is case-sensitive -- the source folders "Images"/"Labels"
   would silently break this).
8. Writes <DATASET_ROOT>/yolo_dataset/data.yaml ready to hand to
   train_yolov8.py.

Run this once before train_yolov8.py. It does not modify your original
data -- it only reads it and writes a cleaned copy elsewhere.
"""

import argparse
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# --------------------------------------------------------------------------
# CONFIG -- edit this if your paths differ
# --------------------------------------------------------------------------
DATASET_ROOT = None
OUTPUT_DIR = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Used only if the automatic filename-based verification (step 6 above)
# can't confidently determine a mapping for some class id.
FALLBACK_CLASS_NAMES = {0: "Aircraft", 1: "Building", 2: "Military Truck", 3: "Tank"}
NUM_CLASSES = 4

# Maps a keyword found in a filename to the "family" that class id belongs to.
# Used purely to verify/derive the class-id -> name mapping from the data.
FILENAME_KEYWORDS = {
    "aircraft": "Aircraft",
    "plane": "Aircraft",
    "building": "Building",
    "tank": "Tank",
    "truck": "Military Truck",
}

DECIMALS = 6  # precision for cleaned label coordinates

# --------------------------------------------------------------------------


def find_split_dirs(root: Path) -> dict:
    """Return {'train': Path, 'val': Path, 'test': Path} for whichever
    splits are present under root."""
    patterns = {
        "train": re.compile(r"train", re.I),
        "val": re.compile(r"valid", re.I),
        "test": re.compile(r"test", re.I),
    }
    found = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name == OUTPUT_DIR.name:
            continue
        for split, pat in patterns.items():
            if pat.search(child.name) and split not in found:
                found[split] = child
    return found


def find_subdir(split_dir: Path, keyword_pat: re.Pattern) -> Path | None:
    for child in split_dir.iterdir():
        if child.is_dir() and keyword_pat.search(child.name):
            return child
    return None


def parse_label_line(line: str, line_no: int, label_path: Path, stats: dict):
    """Parse one YOLO label line, converting polygon format to an
    axis-aligned box if needed. Returns (cls_id, xc, yc, w, h) or None if
    the line should be skipped."""
    parts = line.strip().split()
    if not parts:
        return None

    try:
        cls_id = int(float(parts[0]))
    except ValueError:
        stats["bad_lines"].append(f"{label_path.name}:{line_no} -- non-numeric class id")
        return None

    try:
        coords = [float(p) for p in parts[1:]]
    except ValueError:
        stats["bad_lines"].append(f"{label_path.name}:{line_no} -- non-numeric coordinate")
        return None

    if len(coords) == 4:
        xc, yc, w, h = coords
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        # polygon / OBB: class x1 y1 x2 y2 ... -> convert to axis-aligned box
        xs, ys = coords[0::2], coords[1::2]
        x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
        xc, yc = (x_min + x_max) / 2, (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min
        stats["polygon_converted"] += 1
    else:
        stats["bad_lines"].append(
            f"{label_path.name}:{line_no} -- unexpected field count ({len(coords)} coords)"
        )
        return None

    if cls_id < 0 or cls_id >= NUM_CLASSES:
        stats["bad_lines"].append(f"{label_path.name}:{line_no} -- class id {cls_id} out of range")
        return None

    # clip small float overshoot from the source annotations
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)

    return cls_id, xc, yc, w, h


def process_split(split_name: str, images_dir: Path, labels_dir: Path,
                   out_images: Path, out_labels: Path, stats: dict, class_votes: Counter):
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    label_stems = {p.stem for p in labels_dir.iterdir() if p.suffix.lower() == ".txt"}
    image_stems = {p.stem for p in images}

    orphan_labels = label_stems - image_stems
    for stem in orphan_labels:
        stats["orphan_labels"].append(f"{split_name}/{stem}.txt")

    n_images, n_boxes, n_missing_label = 0, 0, 0
    for img_path in images:
        shutil.copy2(img_path, out_images / img_path.name)
        n_images += 1

        label_path = labels_dir / f"{img_path.stem}.txt"
        out_label_path = out_labels / f"{img_path.stem}.txt"

        if not label_path.exists():
            n_missing_label += 1
            out_label_path.write_text("")  # background image, no objects
            continue

        cleaned = []
        for i, line in enumerate(label_path.read_text().splitlines(), start=1):
            parsed = parse_label_line(line, i, label_path, stats)
            if parsed is None:
                continue
            cls_id, xc, yc, w, h = parsed
            cleaned.append(f"{cls_id} {xc:.{DECIMALS}f} {yc:.{DECIMALS}f} {w:.{DECIMALS}f} {h:.{DECIMALS}f}")
            n_boxes += 1
            stats["class_counts"][cls_id] += 1

            # vote for class-id -> name mapping based on filename prefix
            stem_lower = img_path.stem.lower()
            for kw, family in FILENAME_KEYWORDS.items():
                if stem_lower.startswith(kw):
                    class_votes[(cls_id, family)] += 1
                    break

        out_label_path.write_text("\n".join(cleaned) + ("\n" if cleaned else ""))

    stats["missing_labels"] += n_missing_label
    print(f"  [{split_name}] {n_images} images, {n_boxes} boxes "
          f"({n_missing_label} images had no label file -> treated as background)")


def derive_class_names(class_votes: Counter) -> dict:
    """Pick the best-voted name for each class id; fall back to the
    hardcoded default for any id with no evidence."""
    tally = defaultdict(Counter)
    for (cls_id, family), n in class_votes.items():
        tally[cls_id][family] += n

    names = {}
    for cls_id in range(NUM_CLASSES):
        if tally[cls_id]:
            names[cls_id] = tally[cls_id].most_common(1)[0][0]
        else:
            names[cls_id] = FALLBACK_CLASS_NAMES[cls_id]
    return names


def main():
    global DATASET_ROOT, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Prepare the AE5510 aerial dataset for YOLOv8.")
    parser.add_argument("--root", required=True, type=Path, help="Path to the raw Aerial Image Dataset root.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory; defaults to <root>/yolo_dataset.")
    args = parser.parse_args()
    DATASET_ROOT = args.root
    OUTPUT_DIR = args.output_dir if args.output_dir is not None else DATASET_ROOT / "yolo_dataset"

    if not DATASET_ROOT.exists():
        raise SystemExit(f"DATASET_ROOT does not exist: {DATASET_ROOT}\n"
                          f"Edit the DATASET_ROOT variable at the top of this script.")

    splits = find_split_dirs(DATASET_ROOT)
    if not splits:
        raise SystemExit(f"Could not find any Training/Validation/Testing folders under {DATASET_ROOT}")

    print("Discovered splits:")
    for k, v in splits.items():
        print(f"  {k:5s} -> {v}")

    stats = {
        "polygon_converted": 0,
        "bad_lines": [],
        "orphan_labels": [],
        "missing_labels": 0,
        "class_counts": Counter(),
    }
    class_votes = Counter()

    if OUTPUT_DIR.exists():
        print(f"\nRemoving existing {OUTPUT_DIR} to rebuild it fresh...")
        shutil.rmtree(OUTPUT_DIR)

    print("\nProcessing splits...")
    for split_name, split_dir in splits.items():
        images_dir = find_subdir(split_dir, re.compile("image", re.I))
        labels_dir = find_subdir(split_dir, re.compile("label", re.I))
        if images_dir is None or labels_dir is None:
            print(f"  [{split_name}] SKIPPED -- could not find Images/Labels subfolders in {split_dir}")
            continue
        process_split(
            split_name, images_dir, labels_dir,
            OUTPUT_DIR / "images" / split_name,
            OUTPUT_DIR / "labels" / split_name,
            stats, class_votes,
        )

    class_names = derive_class_names(class_votes)

    data_yaml = {
        "path": str(OUTPUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": {i: class_names[i] for i in range(NUM_CLASSES)},
    }
    if (OUTPUT_DIR / "images" / "test").exists():
        data_yaml["test"] = "images/test"

    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    # ---- report ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Class mapping used (verified against filename prefixes where possible):")
    for i in range(NUM_CLASSES):
        evidence = sum(n for (cid, fam), n in class_votes.items() if cid == i)
        tag = "auto-verified" if evidence else "fallback default, please double check"
        print(f"  {i}: {class_names[i]:<15s} ({tag})")

    print(f"\nPolygon/OBB-format lines converted to axis-aligned boxes: {stats['polygon_converted']}")
    print(f"Images with no label file (treated as background):        {stats['missing_labels']}")
    print(f"Orphan label files (no matching image, skipped):          {len(stats['orphan_labels'])}")
    if stats["orphan_labels"]:
        for x in stats["orphan_labels"][:10]:
            print(f"    - {x}")

    print(f"Malformed lines skipped:                                  {len(stats['bad_lines'])}")
    if stats["bad_lines"]:
        for x in stats["bad_lines"][:10]:
            print(f"    - {x}")

    print("\nBox counts per class (across all processed splits):")
    for i in range(NUM_CLASSES):
        print(f"  {i} ({class_names[i]}): {stats['class_counts'][i]}")

    print(f"\nClean dataset written to: {OUTPUT_DIR}")
    print(f"data.yaml written to:     {yaml_path}")
    print("\nNext step: run train_yolov8.py")


if __name__ == "__main__":
    main()
