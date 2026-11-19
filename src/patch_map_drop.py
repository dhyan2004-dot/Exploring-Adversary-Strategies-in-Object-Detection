"""Full-image metric: drop in Tank AP50 and mAP50 when the patch (or a gray / noise control) is pasted
on every target instance of a split, measured with the Ultralytics validator."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from corruptions import layers_for_condition
from dataset_utils import label_path_for
from patch_evaluate import (ALL_CONDITIONS,CONTROLS,apply_patch_pil,corrupt_pil,detect_training_imgsz,
                            make_control_patch,pick_files_per_photo)

def parse_args():
    p=argparse.ArgumentParser(description="mAP50 drop with the patch on every target-class instance.")
    p.add_argument("--data",required=True)
    p.add_argument("--split",default="val")
    p.add_argument("--weights",required=True)
    p.add_argument("--patch",required=True)
    p.add_argument("--target-class",default="Tank")
    p.add_argument("--patch-scale",type=float,default=0.4)
    p.add_argument("--condition",default="none",choices=ALL_CONDITIONS,
                   help="Corruption applied to the patch tile before pasting")
    p.add_argument("--max-files-per-photo",type=int,default=None)
    p.add_argument("--imgsz",type=int,default=None)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--device",default=None)
    p.add_argument("--output-dir",default="patch_runs/mapdrop")
    p.add_argument("--keep-images",action="store_true",help="Keep the patched copies (default: delete)")
    return p.parse_args()

def build_variant_split(src_images:Path,variant_tile,target_id:int,out_root:Path,files,patch_scale,
                        condition,seed):
    """Write out_root/images/val + labels/val."""
    img_dir,lab_dir=out_root/"images"/"val",out_root/"labels"/"val"
    img_dir.mkdir(parents=True,exist_ok=True)
    lab_dir.mkdir(parents=True,exist_ok=True)
    n_patched=0
    for fi,img_path in enumerate(files):
        lab=label_path_for(img_path)
        lines=[l for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
        im=Image.open(img_path).convert("RGB")
        if variant_tile is not None:
            k=0
            for line in lines:
                parts=line.split()
                if int(parts[0])!=target_id:
                    continue
                box=tuple(map(float,parts[1:5]))
                tile=variant_tile
                if condition!="none":
                    rng=np.random.default_rng((seed*1_000_003+fi)*1009+k)
                    layers=layers_for_condition(condition,tile.size[0],rng)
                    tile=corrupt_pil(tile,layers) if layers else tile
                im=apply_patch_pil(im,tile,box,patch_scale)
                k+=1
                n_patched+=1
        im.save(img_dir/(img_path.stem+".png")) # lossless: no JPEG artefacts around the patch
        (lab_dir/(img_path.stem+".txt")).write_text("\n".join(lines)+("\n" if lines else ""))
    return n_patched

def val_metrics(model,root:Path,names,imgsz,device):
    cfg={"path":str(root),"train":"images/val","val":"images/val","names":names}
    ypath=root/"data.yaml"
    ypath.write_text(yaml.dump(cfg,sort_keys=False))
    r=model.val(data=str(ypath),split="val",imgsz=imgsz,conf=0.001,device=device,plots=False,
                  verbose=False,save_json=False)
    per={names[int(c)]:float(r.box.ap50[i]) for i,c in enumerate(r.box.ap_class_index)}
    return {"mAP50":float(r.box.map50),"AP50_by_class":per}

def main():
    args=parse_args()
    cfg=yaml.safe_load(open(args.data))
    names=cfg["names"]
    name_to_id={v:k for k,v in names.items()}
    target_id=int(args.target_class) if args.target_class.isdigit() else name_to_id[args.target_class]
    target_name=names[target_id]
    split_root=Path(cfg["path"])
    images_dir,labels_dir=split_root/"images"/args.split,split_root/"labels"/args.split
    stems=[f.stem for f in labels_dir.glob("*.txt") if "_dup" not in f.stem]
    if args.max_files_per_photo:
        keep=pick_files_per_photo(stems,args.max_files_per_photo) # same files as patch_evaluate
        stems=[s for s in stems if s in keep]
    files=[]
    for s in sorted(stems):
        c=list(images_dir.glob(f"{s}.*"))
        if c:
            files.append(c[0])
    from ultralytics import YOLO
    model=YOLO(args.weights)
    imgsz=args.imgsz or detect_training_imgsz(args.weights) or 640
    patch=Image.open(args.patch).convert("RGB")
    variants={"clean":None,"adversarial":patch}
    for kind in CONTROLS:
        variants[kind]=make_control_patch(kind,patch.size[0],args.seed)
    out_dir=Path(args.output_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    tmp_root=Path(tempfile.mkdtemp(prefix="mapdrop_",dir=out_dir))
    results={}
    try:
        for name,tile in variants.items():
            root=tmp_root/name
            n=build_variant_split(images_dir,tile,target_id,root,files,args.patch_scale,
                                    args.condition,args.seed)
            results[name]=val_metrics(model,root,names,imgsz,args.device)
            print(name,f"{results[name]['mAP50']:.4f}",
                  f"{results[name]['AP50_by_class'].get(target_name, float('nan')):.4f}",n)
    finally:
        if not args.keep_images:
            shutil.rmtree(tmp_root,ignore_errors=True)
    clean=results["clean"]
    drops={}
    for name in variants:
        if name=="clean":
            continue
        drops[name]={
            "mAP50_drop":clean["mAP50"]-results[name]["mAP50"],
            f"{target_name}_AP50_drop":clean["AP50_by_class"].get(target_name,float("nan"))
                                         -results[name]["AP50_by_class"].get(target_name,float("nan")),
        }
    print(f"\n{'':<14}{'mAP50 drop':>12}{target_name + ' AP50 drop':>20}")
    for name,d in drops.items():
        print(f"  {name:<12}{d['mAP50_drop']:>12.4f}{d[target_name + '_AP50_drop']:>20.4f}")
    summary={"target_class":target_name,"split":args.split,"patch":args.patch,
               "patch_scale":args.patch_scale,"condition":args.condition,"n_files":len(files),
               "max_files_per_photo":args.max_files_per_photo,"imgsz":imgsz,
               "results":results,"drops":drops}
    (out_dir/"mapdrop_summary.json").write_text(json.dumps(summary,indent=2))
    print(out_dir/"mapdrop_summary.json")

if __name__=="__main__":
    main()
