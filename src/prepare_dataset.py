"""Convert the raw Aerial Image Dataset (Training/Validation/Testing, Images/Labels) into a YOLOv8 dataset.
Polygon (9-field) labels become axis-aligned boxes clipped to the image; degenerate boxes are dropped;
writes <DATASET_ROOT>/yolo_dataset/{images,labels}/{train,val,test} and data.yaml.
    python prepare_dataset.py      (DATASET_ROOT defaults to ../data/Aerial Image Dataset)"""

import re
import shutil
from collections import Counter,defaultdict
from pathlib import Path

import yaml

DATASET_ROOT=Path(__file__).resolve().parents[1]/"data"/"Aerial Image Dataset"
OUTPUT_DIR=DATASET_ROOT/"yolo_dataset"

IMAGE_EXTS={".jpg",".jpeg",".png",".bmp"}

FALLBACK_CLASS_NAMES={0:"Aircraft",1:"Building",2:"Military Truck",3:"Tank"}
NUM_CLASSES=4

FILENAME_KEYWORDS={
    "aircraft":"Aircraft",
    "plane":"Aircraft",
    "building":"Building",
    "tank":"Tank",
    "truck":"Military Truck",
}

DECIMALS=6 # precision for cleaned label coordinates
MIN_BOX_FRACTION=0.003

def find_split_dirs(root:Path)->dict:
    """Return {'train': Path."""
    patterns={
        "train":re.compile(r"train",re.I),
        "val":re.compile(r"valid",re.I),
        "test":re.compile(r"test",re.I),
    }
    found={}
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name==OUTPUT_DIR.name:
            continue
        for split,pat in patterns.items():
            if pat.search(child.name) and split not in found:
                found[split]=child
    return found

def find_subdir(split_dir:Path,keyword_pat:re.Pattern)->Path|None:
    for child in split_dir.iterdir():
        if child.is_dir() and keyword_pat.search(child.name):
            return child
    return None

def parse_label_line(line:str,line_no:int,label_path:Path,stats:dict):
    """Parse one YOLO label line, converting polygon format to an axis-aligned box if needed."""
    parts=line.strip().split()
    if not parts:
        return None
    try:
        cls_id=int(float(parts[0]))
    except ValueError:
        stats["bad_lines"].append(f"{label_path.name}:{line_no} -- non-numeric class id")
        return None
    try:
        coords=[float(p) for p in parts[1:]]
    except ValueError:
        stats["bad_lines"].append(f"{label_path.name}:{line_no} -- non-numeric coordinate")
        return None
    if len(coords)==4:
        xc,yc,w,h=coords
    elif len(coords)>=6 and len(coords)%2==0:
        xs,ys=coords[0::2],coords[1::2]
        x_min,x_max,y_min,y_max=min(xs),max(xs),min(ys),max(ys)
        xc,yc=(x_min+x_max)/2,(y_min+y_max)/2
        w,h=x_max-x_min,y_max-y_min
        stats["polygon_converted"]+=1
    else:
        stats["bad_lines"].append(
            f"{label_path.name}:{line_no} -- unexpected field count ({len(coords)} coords)"
        )
        return None
    if cls_id<0 or cls_id>=NUM_CLASSES:
        stats["bad_lines"].append(f"{label_path.name}:{line_no} -- class id {cls_id} out of range")
        return None
    xc=min(max(xc,0.0),1.0)
    yc=min(max(yc,0.0),1.0)
    w=min(max(w,0.0),1.0)
    h=min(max(h,0.0),1.0)
    if w<MIN_BOX_FRACTION or h<MIN_BOX_FRACTION:
        stats["degenerate_boxes"].append(
            f"{label_path.name}:{line_no} -- near-zero box (w={w:.6f}, h={h:.6f}), "
 f"likely a corrupted/collinear source annotation"
        )
        return None
    return cls_id,xc,yc,w,h

def process_split(split_name:str,images_dir:Path,labels_dir:Path,
                   out_images:Path,out_labels:Path,stats:dict,class_votes:Counter):
    out_images.mkdir(parents=True,exist_ok=True)
    out_labels.mkdir(parents=True,exist_ok=True)
    images=sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    label_stems={p.stem for p in labels_dir.iterdir() if p.suffix.lower()==".txt"}
    image_stems={p.stem for p in images}
    orphan_labels=label_stems-image_stems
    for stem in orphan_labels:
        stats["orphan_labels"].append(f"{split_name}/{stem}.txt")
    n_images,n_boxes,n_missing_label=0,0,0
    for img_path in images:
        shutil.copy2(img_path,out_images/img_path.name)
        n_images+=1
        label_path=labels_dir/f"{img_path.stem}.txt"
        out_label_path=out_labels/f"{img_path.stem}.txt"
        if not label_path.exists():
            n_missing_label+=1
            out_label_path.write_text("") # background image, no objects
            continue
        cleaned=[]
        for i,line in enumerate(label_path.read_text().splitlines(),start=1):
            parsed=parse_label_line(line,i,label_path,stats)
            if parsed is None:
                continue
            cls_id,xc,yc,w,h=parsed
            cleaned.append(f"{cls_id} {xc:.{DECIMALS}f} {yc:.{DECIMALS}f} {w:.{DECIMALS}f} {h:.{DECIMALS}f}")
            n_boxes+=1
            stats["class_counts"][cls_id]+=1
            stem_lower=img_path.stem.lower()
            for kw,family in FILENAME_KEYWORDS.items():
                if stem_lower.startswith(kw):
                    class_votes[(cls_id,family)]+=1
                    break
        out_label_path.write_text("\n".join(cleaned)+("\n" if cleaned else ""))
    stats["missing_labels"]+=n_missing_label
    print(split_name,"images",n_images,"boxes",n_boxes,"unlabelled",n_missing_label)

def derive_class_names(class_votes:Counter)->dict:
    """Pick the best-voted name for each class id."""
    tally=defaultdict(Counter)
    for (cls_id,family),n in class_votes.items():
        tally[cls_id][family]+=n
    names={}
    for cls_id in range(NUM_CLASSES):
        if tally[cls_id]:
            names[cls_id]=tally[cls_id].most_common(1)[0][0]
        else:
            names[cls_id]=FALLBACK_CLASS_NAMES[cls_id]
    return names

def main():
    if not DATASET_ROOT.exists():
        raise SystemExit(f"DATASET_ROOT does not exist: {DATASET_ROOT}\n"
 f"Edit the DATASET_ROOT variable at the top of this script.")
    splits=find_split_dirs(DATASET_ROOT)
    if not splits:
        raise SystemExit(f"Could not find any Training/Validation/Testing folders under {DATASET_ROOT}")
    stats={
        "polygon_converted":0,
        "bad_lines":[],
        "degenerate_boxes":[],
        "orphan_labels":[],
        "missing_labels":0,
        "class_counts":Counter(),
    }
    class_votes=Counter()
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for split_name,split_dir in splits.items():
        images_dir=find_subdir(split_dir,re.compile("image",re.I))
        labels_dir=find_subdir(split_dir,re.compile("label",re.I))
        if images_dir is None or labels_dir is None:
            print(split_name,"skipped")
            continue
        process_split(
            split_name,images_dir,labels_dir,
            OUTPUT_DIR/"images"/split_name,
            OUTPUT_DIR/"labels"/split_name,
            stats,class_votes,
        )
    class_names=derive_class_names(class_votes)
    data_yaml={
        "path":str(OUTPUT_DIR),
        "train":"images/train",
        "val":"images/val",
        "names":{i:class_names[i] for i in range(NUM_CLASSES)},
    }
    if (OUTPUT_DIR/"images"/"test").exists():
        data_yaml["test"]="images/test"
    yaml_path=OUTPUT_DIR/"data.yaml"
    with open(yaml_path,"w") as f:
        yaml.dump(data_yaml,f,sort_keys=False)
    for i in range(NUM_CLASSES):
        print(i,class_names[i],stats["class_counts"][i])
    for k in ("polygon_converted","missing_labels"):
        print(k,stats[k])
    for k in ("orphan_labels","bad_lines","degenerate_boxes"):
        print(k,len(stats[k]))
    print(yaml_path)

if __name__=="__main__":
    main()
