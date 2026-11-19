"""Data loading, letterboxing, detector loading (YOLO and RT-DETR) and the CPA per-instance score."""

import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

HERE=Path(__file__).resolve().parent
for extra in (HERE,HERE.parent,HERE.parent/"cv"):
    if extra.exists() and str(extra) not in sys.path:
        sys.path.append(str(extra))

try:
    from dataset_utils import extract_source_group
except ImportError: # minimal fallback (Roboflow hash = photo id)
    import re
    def extract_source_group(stem):
        m=re.search(r"rf\.([0-9a-f]{32})",stem)
        return m.group(1) if m else stem.split(".rf.")[0]

def load_data_cfg(path,target_class):
    cfg=yaml.safe_load(open(path))
    names=cfg["names"]
    if isinstance(names,list):
        names=dict(enumerate(names))
    name_to_id={v:k for k,v in names.items()}
    tid=int(target_class) if str(target_class).isdigit() else name_to_id[target_class]
    return cfg,names,tid

def pick_files_per_photo(stems,k):
    by=defaultdict(list)
    for s in stems:
        by[extract_source_group(s)].append(s)
    keep=set()
    for g,ss in by.items():
        ss=sorted(ss)
        keep.update(ss if len(ss)<=k else random.Random("pick-"+g).sample(ss,k))
    return keep

def list_images_with_target(cfg,split,tid,max_files_per_photo=None):
    """[(image_path, [ (x0,y0,x1,y1) normalised ], [all-boxes...])] for images with >= 1 target."""
    root=Path(cfg["path"])
    img_dir,lab_dir=root/"images"/split,root/"labels"/split
    stems=[f.stem for f in sorted(lab_dir.glob("*.txt")) if "_dup" not in f.stem]
    allowed=pick_files_per_photo(stems,max_files_per_photo) if max_files_per_photo else None
    out=[]
    for stem in stems:
        if allowed is not None and stem not in allowed:
            continue
        cand=[p for p in img_dir.glob(stem+".*")]
        if not cand:
            continue
        boxes=[]
        for line in (lab_dir/f"{stem}.txt").read_text().splitlines():
            p=line.split()
            if len(p)>=5 and int(float(p[0]))==tid:
                if len(p)==5:
                    xc,yc,w,h=map(float,p[1:5])
                    boxes.append((xc-w/2,yc-h/2,xc+w/2,yc+h/2))
                else: # polygon -> axis-aligned box
                    xy=np.array(list(map(float,p[1:])),np.float64).reshape(-1,2)
                    boxes.append((xy[:,0].min(),xy[:,1].min(),xy[:,0].max(),xy[:,1].max()))
        if boxes:
            out.append((cand[0],boxes))
    return out

def letterbox(img_rgb,size):
    """-> float tensor [3,size,size] in [0,1], scale, (padx, pady)."""
    h,w=img_rgb.shape[:2]
    r=size/max(h,w)
    nh,nw=int(round(h*r)),int(round(w*r))
    im=cv2.resize(img_rgb,(nw,nh),interpolation=cv2.INTER_LINEAR) if (nh,nw)!=(h,w) else img_rgb
    canvas=np.full((size,size,3),114,np.uint8)
    py,px=(size-nh)//2,(size-nw)//2
    canvas[py:py+nh,px:px+nw]=im
    return torch.from_numpy(canvas).permute(2,0,1).float()/255.0,r,(px,py),(w,h)

def load_image_and_boxes(path,boxes_norm,size):
    img=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    t,r,(px,py),(w,h)=letterbox(img,size)
    boxes=[(x0*w*r+px,y0*h*r+py,x1*w*r+px,y1*h*r+py) for x0,y0,x1,y1 in boxes_norm]
    return t,boxes

def training_imgsz(weights):
    try:
        ck=torch.load(weights,map_location="cpu",weights_only=False)
        a=ck.get("train_args") or {}
        return int(a["imgsz"]) if "imgsz" in a else None
    except Exception:
        return None

def is_rtdetr(weights):
    try:
        m=torch.load(weights,map_location="cpu",weights_only=False).get("model")
        return m is not None and any(type(x).__name__=="RTDETRDecoder" for x in m.modules())
    except Exception:
        return "rtdetr" in Path(weights).name.lower()

def load_detector(weights):
    """Ultralytics model of the right family (YOLO / RT-DETR need different predictors)."""
    if is_rtdetr(weights):
        from ultralytics import RTDETR
        return RTDETR(weights)
    from ultralytics import YOLO
    return YOLO(weights)

def raw_scores(model,x,tid,nc):
    """Raw head output -> (centres [B,A,2] in input pixels, target-class score [B,A]), differentiable."""
    out=model(x)
    if (isinstance(out,(tuple,list)) and len(out)==2 and isinstance(out[1],(tuple,list))
            and len(out[1])>=2 and torch.is_tensor(out[1][0]) and out[1][0].shape[-1]==4):
        b,sc=out[1][0],out[1][1] # RT-DETR decoder outputs
        b=b.reshape(-1,*b.shape[-2:])[-x.shape[0]:] # [B, Q, 4]
        sc=sc.reshape(-1,*sc.shape[-2:])[-x.shape[0]:].sigmoid() # [B, Q, nc]
        c=b[...,:2]*torch.tensor([x.shape[-1],x.shape[-2]],device=b.device,dtype=b.dtype)
        return c,sc[...,tid]
    y=out[0] if isinstance(out,(tuple,list)) else out
    if y.shape[1]!=4+nc and y.shape[2]==4+nc:
        y=y.transpose(1,2)
    if y.shape[1]!=4+nc:
        raise RuntimeError(f"unexpected detector output {tuple(y.shape)} for nc={nc}")
    return y[:,0:2,:].transpose(1,2),y[:,4+tid,:]

def cpa_instance_score(centres,scores,box,score_thresh=0.05,conf=0.5):
    """CPA code's per-object score: detections of the target class above score_thresh whose centre is inside the..."""
    x0,y0,x1,y1=box
    inside=(centres[:,0]>=x0)&(centres[:,0]<=x1)&(centres[:,1]>=y0)&(centres[:,1]<=y1)
    s=scores[inside&(scores>score_thresh)]
    if s.numel()==0:
        return None
    if s.max()>conf:
        return s[s>conf].mean()
    return s.max()
