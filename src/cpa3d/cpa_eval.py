"""Evaluate 3-D textures on held-out photos on the surrogate and black-box detectors, with gray / noise
sheets of the same size, under none/light/shadow/rain/mud (same draw for every variant).
Stores per-instance confidences before and after for cpa_analyze.py."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from common import (extract_source_group,list_images_with_target,load_data_cfg,load_detector,
                    training_imgsz)
from corrupt_paper import add_flare,add_shadow,add_spatter,flare_params,shadow_params
from place import composite
from render import ViewBank

CONDITIONS=["none","light","shadow","rain","mud"]

def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",required=True)
    p.add_argument("--split",default="val")
    p.add_argument("--target-class",default="Tank")
    p.add_argument("--methods",nargs="+",required=True,
                   help="name=path, path = a cpa_train output dir or a texture .npy")
    p.add_argument("--detectors",nargs="+",required=True,help="name=weights (first = surrogate)")
    p.add_argument("--conditions",default=",".join(CONDITIONS))
    p.add_argument("--mesh",default=str(Path(__file__).with_name("aux_mesh.npz")))
    p.add_argument("--n-views",type=int,default=256)
    p.add_argument("--view-res",type=int,default=128)
    p.add_argument("--elev",type=float,nargs=2,default=(60,90))
    p.add_argument("--patch-m",type=float,default=1.0)
    p.add_argument("--hull-width-m",type=float,default=3.6)
    p.add_argument("--aspect",type=float,default=1.95)
    p.add_argument("--margin-frac",type=float,default=0.05)
    p.add_argument("--iou-match",type=float,default=0.3)
    p.add_argument("--max-files-per-photo",type=int,default=6)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--save-examples",type=int,default=6)
    p.add_argument("--device",default="")
    p.add_argument("--out",required=True)
    return p.parse_args()

def iou(a,b):
    ix=max(0.0,min(a[2],b[2])-max(a[0],b[0]))
    iy=max(0.0,min(a[3],b[3])-max(a[1],b[1]))
    inter=ix*iy
    u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/u if u>0 else 0.0

def match_confs(res,tid,gts,thr_iou):
    out=[0.0]*len(gts)
    if res.boxes is None or len(res.boxes)==0:
        return out
    for c,conf,xyxy in zip(res.boxes.cls.tolist(),res.boxes.conf.tolist(),res.boxes.xyxy.tolist()):
        if int(c)!=tid:
            continue
        for k,g in enumerate(gts):
            if iou(xyxy,g)>=thr_iou:
                out[k]=max(out[k],conf)
    return out

def to_bgr_uint8(t):
    return cv2.cvtColor((t.clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8),
                        cv2.COLOR_RGB2BGR)

def load_texture(path,n_faces):
    """A CPA texture [3, facets] (texture.npy) or an NPA/pixel image patch [3, H, W] (patch.npy)."""
    p=Path(path)
    if p.is_dir():
        p=p/"texture.npy" if (p/"texture.npy").exists() else p/"patch.npy"
    t=np.load(p)
    assert t.shape==(3,n_faces) or (t.ndim==3 and t.shape[0]==3),f"{p}: unexpected patch shape {t.shape}"
    return torch.from_numpy(t).float()

def main():
    a=parse_args()
    device=a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    conds=[c.strip() for c in a.conditions.split(",") if c.strip()]
    assert all(c in CONDITIONS for c in conds),conds
    cfg,names,tid=load_data_cfg(a.data,a.target_class)
    out=Path(a.out);(out/"examples").mkdir(parents=True,exist_ok=True)
    bank=ViewBank(a.mesh,a.n_views,a.view_res,seed=10_000+a.seed,elev=tuple(a.elev),device="cpu",
                    cache=str(out/"viewbank_test.npz"))
    F=bank.n_faces
    variants={}
    for spec in a.methods:
        n,p=spec.split("=",1)
        variants[n]=load_texture(p,F)
    variants["gray"]=torch.full((3,F),0.5)
    variants["noise"]=torch.from_numpy(np.random.default_rng(a.seed).random((3,F))).float()
    dets=[]
    for spec in a.detectors:
        n,w=spec.split("=",1)
        dets.append((n,load_detector(w),training_imgsz(w) or 960))
    items=list_images_with_target(cfg,a.split,tid,a.max_files_per_photo or None)
    print(f"{len(items)} images / {sum(len(b) for _, b in items)} tanks | variants {list(variants)} | "
 f"detectors {[d[0] for d in dets]} | conditions {conds}")
    PK=dict(project=str(out.resolve()/"_ultralytics"),name="predict",exist_ok=True) # no runs/ clutter
    rows={d[0]:[] for d in dets}
    n_ex,t0,inst_id=0,time.time(),0
    for ii,(path,boxes_n) in enumerate(items):
        rgb0=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        H,W=rgb0.shape[:2]
        base=torch.from_numpy(rgb0).permute(2,0,1).float()/255
        gts=[(x0*W,y0*H,x1*W,y1*H) for x0,y0,x1,y1 in boxes_n]
        rng=np.random.default_rng(a.seed*1_000_003+ii)
        views=[(int(rng.integers(len(bank))),int(rng.choice([-1,1]))) for _ in gts]
        before={n:match_confs(m.predict(to_bgr_uint8(base),imgsz=sz,conf=0.001,device=device,
                                           verbose=False,**PK)[0],tid,gts,a.iou_match) for n,m,sz in dets}
        recs=[{"instance":inst_id+k,"image":path.name,"group":extract_source_group(path.stem),
                 "box":[round(v,1) for v in g]} for k,g in enumerate(gts)]
        per_det={n:[dict(r,before_conf=before[n][k]) for k,r in enumerate(recs)] for n,_,_ in dets}
        for ci,cond in enumerate(conds):
            crng_seed=a.seed*7_919+ii*31+ci
            for vname,tex in variants.items():
                img,mask_all=base,[]
                for (vi,sg),g in zip(views,gts):
                    r_,al=bank.render(tex,vi)
                    img,m=composite(img,g,r_,al,a.patch_m,a.hull_width_m,a.aspect,a.margin_frac,sign=sg)
                    mask_all.append(m)
                crng=np.random.default_rng(crng_seed) # same draw for every variant
                if cond=="light":
                    for m in mask_all:
                        img=add_flare(img,flare_params(m.numpy(),crng))
                elif cond=="shadow":
                    for g in gts:
                        img=add_shadow(img,[shadow_params(g,crng)])
                elif cond in ("rain","mud"):
                    img=add_spatter(img,cond,crng)
                bgr=to_bgr_uint8(img)
                for n,m,sz in dets:
                    res=m.predict(bgr,imgsz=sz,conf=0.001,device=device,verbose=False,**PK)[0]
                    for k,c in enumerate(match_confs(res,tid,gts,a.iou_match)):
                        per_det[n][k][f"after_{cond}_{vname}"]=c
                    if (n==dets[0][0] and cond=="none" and vname==next(iter(variants))
                            and n_ex<a.save_examples and max(before[n])>=0.5):
                        b_show=m.predict(to_bgr_uint8(base),imgsz=sz,conf=0.5,device=device,verbose=False,**PK)[0].plot()
                        a_show=m.predict(bgr,imgsz=sz,conf=0.5,device=device,verbose=False,**PK)[0].plot()
                        cv2.imwrite(str(out/"examples"/f"{path.stem}.jpg"),np.concatenate([b_show,a_show],1))
                        n_ex+=1
        for n in per_det:
            rows[n].extend(per_det[n])
        inst_id+=len(gts)
        if ii%10==0:
            print(f"  {ii + 1}/{len(items)} images  {time.time() - t0:.0f}s",flush=True)
    for n,_,sz in dets:
        d=out/n
        d.mkdir(exist_ok=True)
        json.dump(rows[n],open(d/"per_instance.json","w"))
        r=rows[n]
        summ={"detector":n,"imgsz":sz,"n_instances":len(r),"variants":list(variants),"conditions":conds,
                "iou_match":a.iou_match,"config":vars(a)}
        for thr in (0.5,0.25):
            det=[x for x in r if x["before_conf"]>=thr]
            summ[f"asr@{thr}"]={"n_detected_before":len(det),
                                  **{f"{c}/{v}":(sum(x[f"after_{c}_{v}"]<thr for x in det)/len(det) if det else None)
                                     for c in conds for v in variants}}
        json.dump(summ,open(d/"summary.json","w"),indent=2)
        s=summ["asr@0.5"]
        print(f"[{n}] detected before (conf>=0.5): {s['n_detected_before']}/{len(r)}  ASR none: "+
              "  ".join(f"{v} {s[f'none/{v}']:.2f}" for v in variants if s.get(f'none/{v}') is not None))
    json.dump({"done":True},open(out/"stage_done.json","w"))

if __name__=="__main__":
    main()
