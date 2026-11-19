"""K-fold cross-validation grouped by source photo (no photo on both sides). Builds <root>/cv/fold*/
with hard links, data.yaml and folds.json; rare classes are oversampled in the training part only.
    python make_group_split.py --root <yolo_dataset> --folds 5"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from dataset_utils import extract_source_group,label_path_for

IMG_EXT={".jpg",".jpeg",".png",".bmp"}

def parse_args():
    p=argparse.ArgumentParser(description="Photo-grouped K-fold split.")
    p.add_argument("--root",required=True,help="The yolo_dataset folder (contains images/, labels/, data.yaml).")
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--pool-splits",type=str,default="train,dev_val",
                   help="Subfolders of images/ that form the pool to split.")
    p.add_argument("--oversample",type=str,default="Military Truck=3,Tank=2",
                   help="Per-class duplication factor for TRAINING files (a file gets the largest factor "
 "of the classes it contains). Same as balance_dataset.py. Use '' to disable.")
    p.add_argument("--force",action="store_true",help="Delete and rebuild <root>/cv if it exists.")
    return p.parse_args()

def read_classes(label_path:Path):
    """{class_id: n_instances} for one label file."""
    counts=defaultdict(int)
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])]+=1
    return counts

def assign_folds(group_counts,k,seed):
    """group_counts: {group: {class_id: n_instances}} -> {group: fold}."""
    rng=random.Random(seed)
    photos_per_class=defaultdict(int)
    for cc in group_counts.values():
        for c in cc:
            photos_per_class[c]+=1
    class_order=sorted(photos_per_class,key= lambda c:(photos_per_class[c],c))
    fold_of={}
    f_photos=[defaultdict(int) for _ in range(k)] # [fold][class] -> photos
    f_inst=[defaultdict(int) for _ in range(k)] # [fold][class] -> instances
    f_total=[0]*k
    def put(g,f):
        fold_of[g]=f
        f_total[f]+=1
        for c,n in group_counts[g].items():
            f_photos[f][c]+=1
            f_inst[f][c]+=n
    for c in class_order:
        todo=sorted(g for g,cc in group_counts.items() if c in cc and g not in fold_of)
        rng.shuffle(todo)
        for g in todo:
            f=min(range(k),key= lambda i:(f_photos[i][c],f_inst[i][c],f_total[i],i))
            put(g,f)
    rest=sorted(g for g in group_counts if g not in fold_of) # photos with no objects at all
    rng.shuffle(rest)
    for g in rest:
        put(g,min(range(k),key= lambda i:(f_total[i],i)))
    return fold_of

def link(src:Path,dst:Path):
    try:
        os.link(src,dst)
    except OSError:
        shutil.copy2(src,dst)

def main():
    args=parse_args()
    root=Path(args.root).resolve()
    cfg=yaml.safe_load(open(root/"data.yaml"))
    names=cfg["names"]
    cv_dir=root/"cv"
    if cv_dir.exists():
        if (cv_dir/"folds.json").exists() and not args.force:
            raise SystemExit(f"{cv_dir} already built. Pass --force to rebuild it "
 f"(only do that before any cv_run.py results exist).")
        if not (cv_dir/"folds.json").exists():
            print(f"{cv_dir} exists but has no folds.json (an earlier build was interrupted) -- rebuilding.")
        shutil.rmtree(cv_dir)
    factors={}
    name_to_id={v:k for k,v in names.items()}
    for item in [x for x in args.oversample.split(",") if x.strip()]:
        nm,fac=item.rsplit("=",1)
        if nm.strip() not in name_to_id:
            raise SystemExit(f"--oversample: unknown class '{nm.strip()}' (names: {list(names.values())})")
        factors[name_to_id[nm.strip()]]=int(fac)
    files=[] # (img_path, group, is_dup)
    for split in [s.strip() for s in args.pool_splits.split(",") if s.strip()]:
        d=root/"images"/split
        if not d.exists():
            print(f"(skipping missing pool split {d})")
            continue
        for img in sorted(d.iterdir()):
            if img.suffix.lower() in IMG_EXT and "_dup" not in img.stem:
                files.append((img,extract_source_group(img.stem),False))
    if not files:
        raise SystemExit("No images found in the pool.")
    group_counts=defaultdict(lambda:defaultdict(int))
    for img,g,is_dup in files:
        if is_dup:
            continue
        for c,n in read_classes(label_path_for(img)).items():
            group_counts[g][c]+=n
        group_counts[g] # touch so label-less photos still exist as groups
    for _,g,_ in files:
        group_counts[g]
    group_counts={g:dict(cc) for g,cc in group_counts.items()}
    k=args.folds
    if len(group_counts)<k:
        raise SystemExit(f"Only {len(group_counts)} photos, cannot make {k} folds.")
    fold_of=assign_folds(group_counts,k,args.seed)
    print(f"Pool: {len(files)} original files from {len(group_counts)} distinct source photos -> {k} folds")
    print(f"Oversampling in each fold's TRAIN part only: { {names[c]: f for c, f in factors.items()} }")
    for split in ("val","test"):
        d=root/"images"/split
        if d.exists():
            other={extract_source_group(p.stem) for p in d.iterdir() if p.suffix.lower() in IMG_EXT}
            shared=other&set(group_counts)
            print(f"  official {split}: {len(other)} photos, {len(shared)} shared with the pool"
 f"{'  <-- LEAK, investigate' if shared else ''}")
    by_group=defaultdict(list)
    for img,g,is_dup in files:
        by_group[g].append((img,is_dup))
    file_factor={}
    for img,_,_ in files:
        cls=read_classes(label_path_for(img))
        file_factor[img]=max([factors.get(c,1) for c in cls]+[1])
    info={}
    test_dir=root/"images"/"test"
    for f in range(k):
        fd=cv_dir/f"fold{f}"
        for sub in ("images/train","images/val","labels/train","labels/val"):
            (fd/sub).mkdir(parents=True,exist_ok=True)
        n_tr=n_va=0
        for g,members in by_group.items():
            held=fold_of[g]==f
            for img,is_dup in members:
                if held and is_dup:
                    continue # never put oversampling copies in held-out
                sub="val" if held else "train"
                lab=label_path_for(img)
                copies=1 if held else file_factor[img]
                for c in range(copies):
                    name=img.stem if c==0 else f"{img.stem}_dup{c}"
                    link(img,fd/"images"/sub/(name+img.suffix))
                    if lab.exists():
                        shutil.copy2(lab,fd/"labels"/sub/(name+".txt"))
                    n_va+=held
                    n_tr+= not held
        fold_cfg={"path":str(fd),"train":"images/train","val":"images/val","names":names}
        if test_dir.exists():
            fold_cfg["test"]=str(test_dir)
        with open(fd/"data.yaml","w") as fh:
            yaml.dump(fold_cfg,fh,sort_keys=False)
        held_groups=sorted(g for g in group_counts if fold_of[g]==f)
        tr_groups=sorted(g for g in group_counts if fold_of[g]!=f)
        assert not set(held_groups)&set(tr_groups)
        tr_g={extract_source_group(p.stem) for p in (fd/"images/train").iterdir()}
        va_g={extract_source_group(p.stem) for p in (fd/"images/val").iterdir()}
        assert not tr_g&va_g,"photo appears in both train and val!"
        per_class={}
        for c,nm in names.items():
            per_class[nm]={
                "heldout_photos":sum(1 for g in held_groups if c in group_counts[g]),
                "heldout_instances":sum(group_counts[g].get(c,0) for g in held_groups),
                "train_photos":sum(1 for g in tr_groups if c in group_counts[g]),
            }
        info[f"fold{f}"]={"train_files":n_tr,"val_files":n_va,"heldout_photos":held_groups,
                            "per_class":per_class}
    with open(cv_dir/"folds.json","w") as fh:
        json.dump({"seed":args.seed,"k":k,"oversample":args.oversample,
                   "group_to_fold":fold_of,"folds":info},fh,indent=1)
    print(f"\n{'fold':<7}{'train files':>12}{'held-out files':>16}{'held-out photos':>17}   "
          +"  ".join(f"{nm[:10]:>10}" for nm in names.values()))
    print("        (class columns: held-out photos / held-out instances)")
    for f in range(k):
        r=info[f"fold{f}"]
        cols="  ".join(f"{r['per_class'][nm]['heldout_photos']:>4}/{r['per_class'][nm]['heldout_instances']:<5}"
                         for nm in names.values())
        print(f"fold{f:<3}{r['train_files']:>12}{r['val_files']:>16}{len(r['heldout_photos']):>17}   {cols}")
    tank=[nm for nm in names.values() if nm=="Tank"]
    for nm in tank:
        mn=min(info[f"fold{f}"]["per_class"][nm]["heldout_photos"] for f in range(k))
        print(f"\nSmallest held-out {nm} photo count in any fold: {mn}")
    print(f"\nWrote {cv_dir}  (data.yaml per fold, folds.json). No photo is in both train and val of any fold.")

if __name__=="__main__":
    main()
