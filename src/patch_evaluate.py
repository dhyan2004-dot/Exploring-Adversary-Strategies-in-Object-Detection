"""Per-instance evaluation of a 2-D patch with the full pipeline (NMS): a target detected before is
'hidden' if its confidence falls below the threshold. Gray / noise squares of the same size and place are
controls; conditions none/light/shadow/rain/mud/mixed use the same random draw for every method."""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image,ImageDraw

from corruptions import apply_layers,layers_for_condition
from dataset_utils import extract_source_group
from photo_stats import cluster_bootstrap

DATASET_ROOT=Path(__file__).resolve().parents[1]/"data"/"Aerial Image Dataset"
DEFAULT_DATA_YAML=DATASET_ROOT/"yolo_dataset"/"data.yaml"

ALL_CONDITIONS=["none","light","shadow","rain","mud","mixed"]
CONTROLS=["gray","noise"]

def parse_args():
    p=argparse.ArgumentParser(description="Evaluate a patch's attack effectiveness (with controls).")
    p.add_argument("--data",type=str,default=str(DEFAULT_DATA_YAML))
    p.add_argument("--weights",type=str,required=True)
    p.add_argument("--patch",type=str,required=True,help="Path to the patch PNG.")
    p.add_argument("--target-class",type=str,default="Tank")
    p.add_argument("--split",type=str,default="dev_val",choices=["train","val","test","dev_val"])
    p.add_argument("--patch-scale",type=float,default=0.5,
                    help="Patch width as a fraction of the object's box width. Use the value the patch was trained with.")
    p.add_argument("--corruptions",type=str,default="none",
                    help="Comma-separated conditions from: none,light,shadow,rain,mud,mixed -- or 'all'.")
    p.add_argument("--conf-threshold",type=float,default=0.25,
                    help="Confidence at or above which the object counts as detected.")
    p.add_argument("--iou-match",type=float,default=0.3,
                    help="IoU with the ground-truth box needed for a prediction to count as that object.")
    p.add_argument("--imgsz",type=int,default=None,
                    help="Inference resolution; must match the detector's training imgsz. Auto-detected "
 "from the checkpoint if omitted (otherwise Ultralytics silently uses 640).")
    p.add_argument("--device",type=str,default=None)
    p.add_argument("--seed",type=int,default=0,help="Seed for corruption sampling and the noise control.")
    p.add_argument("--max-instances",type=int,default=None)
    p.add_argument("--max-files-per-photo",type=int,default=None,
                    help="Each source photo has up to ~36 augmented variants. Evaluate at most this many per "
 "photo (chosen deterministically, so every patch/method sees the SAME files). "
 "6 is a good value: it keeps lighting/zoom variety and cuts run time ~6x. "
 "Photos, not files, are the statistical unit either way.")
    p.add_argument("--save-examples",type=int,default=8,
                    help="Before/after image pairs to save (half successes, half failures, one per source photo).")
    p.add_argument("--output-dir",type=str,default="patch_runs/eval")
    return p.parse_args()

def detect_training_imgsz(weights_path):
    try:
        import torch
        ckpt=torch.load(weights_path,map_location="cpu",weights_only=False)
        train_args=ckpt.get("train_args",{})
        imgsz=train_args.get("imgsz") if isinstance(train_args,dict) else getattr(train_args,"imgsz",None)
        return int(imgsz) if imgsz is not None else None
    except Exception:
        return None

def make_control_patch(kind:str,size:int,seed:int=0):
    """Non-adversarial patches of the same size: separate 'the adversarial pattern works' from 'any square..."""
    if kind=="noise":
        arr=np.random.default_rng(seed).integers(0,256,(size,size,3),dtype=np.uint8)
    else:
        arr=np.full((size,size,3),128,dtype=np.uint8)
    return Image.fromarray(arr)

def wilson_interval(k:int,n:int,z:float=1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n==0:
        return None,None
    p=k/n
    denom=1+z*z/n
    centre=(p+z*z/(2*n))/denom
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return max(0.0,centre-half),min(1.0,centre+half)

def iou_xyxy(box_a,box_b):
    ax0,ay0,ax1,ay1=box_a
    bx0,by0,bx1,by1=box_b
    ix0,iy0=max(ax0,bx0),max(ay0,by0)
    ix1,iy1=min(ax1,bx1),min(ay1,by1)
    iw,ih=max(0.0,ix1-ix0),max(0.0,iy1-iy0)
    inter=iw*ih
    area_a=max(0.0,ax1-ax0)*max(0.0,ay1-ay0)
    area_b=max(0.0,bx1-bx0)*max(0.0,by1-by0)
    union=area_a+area_b-inter
    return inter/union if union>0 else 0.0

def pick_files_per_photo(stems,k):
    """Deterministic: at most k files per source photo (same choice on every run / every method)."""
    by=defaultdict(list)
    for s in stems:
        by[extract_source_group(s)].append(s)
    keep=set()
    for g,ss in by.items():
        ss=sorted(ss)
        keep.update(ss if len(ss)<=k else random.Random("pick-"+g).sample(ss,k))
    return keep

def load_target_class_samples(images_dir:Path,labels_dir:Path,target_id:int,max_instances=None,
                              max_files_per_photo=None):
    samples=[] # (image_path, (xc,yc,w,h))
    allowed=None
    if max_files_per_photo:
        allowed=pick_files_per_photo([f.stem for f in labels_dir.glob("*.txt") if "_dup" not in f.stem],
                                       max_files_per_photo)
    for label_file in sorted(labels_dir.glob("*.txt")):
        if "_dup" in label_file.stem:
            continue
        if allowed is not None and label_file.stem not in allowed:
            continue
        candidates=list(images_dir.glob(f"{label_file.stem}.*"))
        if not candidates:
            continue
        img_path=candidates[0]
        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue
            parts=line.split()
            if int(parts[0])!=target_id:
                continue
            samples.append((img_path,tuple(map(float,parts[1:5]))))
            if max_instances and len(samples)>=max_instances:
                return samples
    return samples

def apply_patch_pil(image:Image.Image,patch:Image.Image,box,patch_scale:float):
    """Paste the patch centred on the box, sized relative to the box width."""
    img=image.copy()
    W,H=img.size
    xc,yc,w,h=box
    place_size=max(4,int(round(w*W*patch_scale)))
    resized=patch.resize((place_size,place_size),Image.BILINEAR)
    img.paste(resized,(int(round(xc*W-place_size/2)),int(round(yc*H-place_size/2))))
    return img

def corrupt_pil(patch:Image.Image,layers):
    """Apply corruption layers to a PIL patch tile and return a PIL image."""
    arr=np.asarray(patch,dtype=np.float32).transpose(2,0,1)/255.0
    out=apply_layers(arr,layers)
    return Image.fromarray((np.clip(out,0,1).transpose(1,2,0)*255).round().astype(np.uint8))

def best_match_confidence(results,target_id:int,gt_xyxy,iou_thresh:float):
    """Confidence of whichever predicted box of target_id best matches gt_xyxy (0.0 if none)."""
    boxes=results.boxes
    if boxes is None or len(boxes)==0:
        return 0.0
    best=0.0
    for cls_id,conf,xyxy in zip(boxes.cls.tolist(),boxes.conf.tolist(),boxes.xyxy.tolist()):
        if int(cls_id)==target_id and iou_xyxy(xyxy,gt_xyxy)>=iou_thresh:
            best=max(best,conf)
    return best

def summarize(before,after,detected,thr):
    n_det=int(detected.sum())
    if n_det==0:
        return None
    gone=int((after[detected]<thr).sum())
    lo,hi=wilson_interval(gone,n_det)
    return {
        "disappeared":gone,
        "n_detected_before":n_det,
        "disappearance_rate":gone/n_det,
        "disappearance_ci95":[lo,hi],
        "mean_confidence_after":float(after[detected].mean()),
        "mean_confidence_drop":float((before[detected]-after[detected]).mean()),
    }

def save_example(path,before_res,after_res,gt_xyxy,before_conf,after_conf):
    panels=[]
    for res,text in ((before_res,f"BEFORE: attacked-object conf {before_conf:.2f}"),
                      (after_res,f"AFTER:  attacked-object conf {after_conf:.2f}")):
        im=Image.fromarray(res.plot()[:,:,::-1]) # plot() is BGR
        d=ImageDraw.Draw(im)
        d.rectangle(gt_xyxy,outline=(255,255,0),width=3)
        d.rectangle([0,0,im.width,16],fill=(0,0,0))
        d.text((6,3),text+"   (yellow box = attacked instance)",fill=(255,255,0))
        panels.append(im)
    combo=Image.new("RGB",(panels[0].width*2+10,panels[0].height),"white")
    combo.paste(panels[0],(0,0))
    combo.paste(panels[1],(panels[0].width+10,0))
    combo.save(path)

def main():
    args=parse_args()
    conditions=ALL_CONDITIONS if args.corruptions=="all" else [c.strip() for c in args.corruptions.split(",")]
    bad=[c for c in conditions if c not in ALL_CONDITIONS]
    if bad:
        raise SystemExit(f"Unknown --corruptions value(s) {bad}; choose from {ALL_CONDITIONS} or 'all'.")
    with open(args.data) as f:
        data_cfg=yaml.safe_load(f)
    names=data_cfg["names"]
    name_to_id={v:k for k,v in names.items()}
    target_id=int(args.target_class) if args.target_class.isdigit() else name_to_id[args.target_class]
    print(f"Target class: {names[target_id]} (id={target_id})")
    from ultralytics import YOLO
    model=YOLO(args.weights)
    imgsz=args.imgsz or detect_training_imgsz(args.weights)
    if imgsz is None:
        imgsz=640
        print("WARNING: could not auto-detect training imgsz; using 640. Pass --imgsz explicitly if wrong.")
    else:
        print(f"Using inference imgsz={imgsz}")
    patch=Image.open(args.patch).convert("RGB")
    variants={"adversarial":patch}
    for kind in CONTROLS:
        variants[kind]=make_control_patch(kind,patch.size[0],args.seed)
    dataset_root=Path(data_cfg["path"])
    images_dir=dataset_root/"images"/args.split
    labels_dir=dataset_root/"labels"/args.split
    samples=load_target_class_samples(images_dir,labels_dir,target_id,args.max_instances,
                                        args.max_files_per_photo)
    if not samples:
        raise SystemExit(f"No instances of '{names[target_id]}' found in {images_dir}")
    print(f"Evaluating {len(samples)} instances of '{names[target_id]}' from '{args.split}' "
 f"| patch_scale={args.patch_scale} | conditions: {conditions}")
    out_dir=Path(args.output_dir)
    examples_dir=out_dir/"examples"
    examples_dir.mkdir(parents=True,exist_ok=True)
    thr=args.conf_threshold
    before_confs=[]
    after_confs={c:{v:[] for v in variants} for c in conditions}
    per_instance=[]
    seen_groups=set()
    example_counts={"success":0,"fail":0}
    example_quota=(args.save_examples+1)//2
    for idx,(img_path,box) in enumerate(samples):
        orig=Image.open(img_path).convert("RGB")
        W,H=orig.size
        xc,yc,w,h=box
        gt_xyxy=((xc-w/2)*W,(yc-h/2)*H,(xc+w/2)*W,(yc+h/2)*H)
        before_res=model.predict(orig,conf=0.001,imgsz=imgsz,device=args.device,verbose=False)[0]
        before_conf=best_match_confidence(before_res,target_id,gt_xyxy,args.iou_match)
        before_confs.append(before_conf)
        row={"instance":idx,"image":img_path.name,"group":extract_source_group(img_path.stem),
               "before_conf":before_conf}
        for cond in conditions:
            rng=np.random.default_rng(args.seed*1_000_003+idx)
            layers=layers_for_condition(cond,patch.size[0],rng)
            for name,tile in variants.items():
                tile_c=corrupt_pil(tile,layers) if layers else tile
                patched=apply_patch_pil(orig,tile_c,box,args.patch_scale)
                res=model.predict(patched,conf=0.001,imgsz=imgsz,device=args.device,verbose=False)[0]
                c=best_match_confidence(res,target_id,gt_xyxy,args.iou_match)
                after_confs[cond][name].append(c)
                row[f"after_{cond}_{name}"]=c
                if cond==conditions[0] and name=="adversarial" and before_conf>=thr:
                    group=extract_source_group(img_path.stem)
                    kind="success" if c<thr else "fail"
                    if group not in seen_groups and example_counts[kind]<example_quota:
                        b_show=model.predict(orig,conf=thr,imgsz=imgsz,device=args.device,verbose=False)[0]
                        a_show=model.predict(patched,conf=thr,imgsz=imgsz,device=args.device,verbose=False)[0]
                        save_example(examples_dir/f"{kind}_{img_path.stem}.jpg",
                                     b_show,a_show,gt_xyxy,before_conf,c)
                        seen_groups.add(group)
                        example_counts[kind]+=1
        per_instance.append(row)
    before=np.array(before_confs)
    detected=before>=thr
    n_det=int(detected.sum())
    results={cond:{v:summarize(before,np.array(after_confs[cond][v]),detected,thr) for v in variants}
               for cond in conditions}
    photo_level={}
    if n_det:
        groups=[row["group"] for row,d in zip(per_instance,detected) if d]
        for cond in conditions:
            series={v:(np.array(after_confs[cond][v])[detected]<thr).astype(float) for v in variants}
            photo_level[cond]=cluster_bootstrap(groups,series)
    summary={
        "target_class":names[target_id],"split":args.split,"patch":str(args.patch),
        "patch_scale":args.patch_scale,"n_instances_evaluated":len(samples),
        "n_detected_before_patch":n_det,
        "mean_confidence_before":float(before[detected].mean()) if n_det else None,
        "conf_threshold":thr,"iou_match_threshold":args.iou_match,"imgsz":imgsz,"seed":args.seed,
        "n_photos_with_detected_instances":len({row["group"] for row,d in zip(per_instance,detected) if d}),
        "max_files_per_photo":args.max_files_per_photo,
        "results_by_condition":results,
        "photo_level_by_condition":photo_level,
    }
    print("\n"+"="*78)
    print(f"ATTACK EFFECTIVENESS  |  {args.patch}  |  {args.split}  |  patch_scale={args.patch_scale}")
    print("="*78)
    mean_before=before[detected].mean() if n_det else float("nan")
    print(f"  Detected before patching: {n_det} of {len(samples)} instances "
 f"(conf >= {thr}, IoU >= {args.iou_match});  mean confidence {mean_before:.3f}")
    if n_det:
        print(f"\n  Share of detected instances that DISAPPEAR (conf < {thr}) after the patch:")
        print(f"  {'condition':<12}{'ADVERSARIAL  [95% CI]':>28}{'gray ctrl':>12}{'noise ctrl':>12}")
        for cond in conditions:
            r=results[cond]
            a=r["adversarial"]
            lo,hi=a["disappearance_ci95"]
            print(f"  {cond:<12}{a['disappearance_rate']:>10.2f}  [{lo:.2f}, {hi:.2f}]"
 f"{r['gray']['disappearance_rate']:>12.2f}{r['noise']['disappearance_rate']:>12.2f}")
        print("\n  The attack counts only for what it achieves BEYOND the controls (same size, same place).")
        print("  The Wilson CIs above treat instances as independent -> too narrow. Honest version "
 "(photos resampled):")
        print(f"  {'condition':<12}{'adversarial [95% CI]':>26}{'margin over best control [95% CI]':>40}")
        for cond in conditions:
            pl=photo_level[cond]
            lo,hi=pl["ci95"]["adversarial"]
            mlo,mhi=pl["margin_ci95"]
            print(f"  {cond:<12}{pl['rate']['adversarial']:>9.2f}  [{lo:.2f}, {hi:.2f}]"
 f"{pl['margin_over_best_control']:>+16.2f}  [{mlo:+.2f}, {mhi:+.2f}]")
        n_ph=summary["n_photos_with_detected_instances"]
        print(f"  ({n_ph} distinct photos contribute; a margin whose interval includes 0 is not evidence "
 f"that the pattern beats plain occlusion.)")
    if n_det<100:
        print(f"\n  WARNING: only {n_det} instances were detected before patching -- rates are very uncertain.")
    if n_det<0.3*len(samples):
        print("  WARNING: the detector finds under 30% of these instances even without a patch -- check "
 "imgsz/colour handling before trusting this.")
    with open(out_dir/"attack_summary.json","w") as f:
        json.dump(summary,f,indent=2)
    with open(out_dir/"per_instance.json","w") as f:
        json.dump(per_instance,f,indent=2)
    print(f"\nSummary: {out_dir / 'attack_summary.json'}")
    print(f"Examples ({example_counts['success']} success / {example_counts['fail']} fail, "
 f"one per source photo): {examples_dir}")

if __name__=="__main__":
    main()
