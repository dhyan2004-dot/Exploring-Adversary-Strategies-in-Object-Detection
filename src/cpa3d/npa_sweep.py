"""Realism-vs-attack curve without retraining: for each saturation budget S pick the last saved epoch of
the npa_stop run with <=S% saturated pixels.
    python npa_sweep.py --runs cpa3d_runs --fold 0 --budgets 5 10 20 40 100"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

def sat(p):
    return float(((p<0.03)|(p>0.97)).any(0).mean()*100)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs",default="cpa3d_runs")
    ap.add_argument("--fold",type=int,required=True)
    ap.add_argument("--source",default="npa_stop")
    ap.add_argument("--budgets",type=float,nargs="+",default=[5,10,20,40,100])
    a=ap.parse_args()
    src=Path(a.runs)/f"fold{a.fold}"/"textures"/a.source
    pngs=sorted(src.glob("examples/patch_epoch*.png"))
    if not pngs:
        raise SystemExit(f"no per-epoch patches in {src}/examples")
    eps=[]
    for f in pngs:
        p=cv2.cvtColor(cv2.imread(str(f)),cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2,0,1)/255
        eps.append((int(f.stem.replace("patch_epoch","")),p,sat(p)))
    init=np.load(src/"patch_init.npy")
    cfg=json.load(open(src/"config.json"))
    print(f"fold {a.fold}: saturation per epoch:"," ".join(f"{e}:{s:.0f}%" for e,_,s in eps))
    for S in a.budgets:
        ok=[x for x in eps if x[2]<=S]
        ep,patch,s=ok[-1] if ok else (-1,init,sat(init))
        name=f"npa_stop{int(S)}"
        d=src.parent/name
        d.mkdir(exist_ok=True)
        np.save(d/"patch.npy",patch.astype(np.float32))
        np.save(d/"patch_init.npy",init)
        json.dump(cfg|{"natural_stop":S,"selected_epoch":ep,"selected_saturation":s,"derived_from":a.source},
                  open(d/"config.json","w"),indent=2)
        json.dump({"done":True},open(d/"stage_done.json","w"))
        print(f"  {name}: epoch {ep} ({s:.1f}% saturated)")

if __name__=="__main__":
    main()
