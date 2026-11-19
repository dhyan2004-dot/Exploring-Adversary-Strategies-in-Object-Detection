"""Naturalness proxies for GAN / camouflage / pixel patches (mean over folds): final latent l-inf,
% saturated pixels (start -> final), drift from the start image; writes a start|final image sheet."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

def sat(p):
    return float(((p<0.03)|(p>0.97)).any(0).mean()*100)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs",default="cpa3d_runs")
    a=ap.parse_args()
    root=Path(a.runs)
    by=defaultdict(dict) # variant -> fold -> (init, final, z_inf)
    for d in sorted(root.glob("fold*/textures/*")):
        if not (d.name=="pixel" or d.name.startswith(("npa","camo"))) or "__" in d.name:
            continue
        if not (d/"patch.npy").exists() or not (d/"stage_done.json").exists():
            continue
        fin=np.load(d/"patch.npy")
        ini=np.load(d/"patch_init.npy") if (d/"patch_init.npy").exists() else None
        zi=float(np.abs(np.load(d/"latent.npy")).max()) if (d/"latent.npy").exists() else float("nan")
        by[d.name][d.parent.parent.name]=(ini,fin,zi)
    if not by:
        raise SystemExit("no NPA patches found")
    def label(v):
        if v=="pixel":
            return "pixel (no GAN)"
        if v=="npa":
            return "npa  class 259, tau 10 (main run)"
        cfg=json.load(open(next(root.glob(f"fold*/textures/{v}/config.json"))))
        if cfg.get("mode")=="camo":
            return f"{v}  camouflage, palette {cfg['palette']}, {cfg.get('patch_m', 1.0):g} m"
        extra=", z on sphere" if cfg.get("z_proj")=="sphere" else ""
        extra+=f", chunks {cfg['z_chunks']}" if cfg.get("z_chunks") else ""
        extra+=f", sat-penalty {cfg['sat_weight']:g}" if cfg.get("sat_weight") else ""
        extra+=f", natural-stop {cfg['natural_stop']:g}%" if cfg.get("natural_stop") else ""
        return f"{v}  class {cfg['gan_class']}, tau {cfg['tau']:g}{extra}"
    lines=["# NPA naturalness proxies (mean over folds)\n",
             "| patch | folds | final latent l-inf | saturated pixels % (start -> final) | drift from start |",
             "|---|---|---|---|---|"]
    order=sorted(by,key= lambda v:(v=="pixel",v.startswith("camo"),v!="npa",v))
    for v in order:
        f=by[v]
        z=np.nanmean([x[2] for x in f.values()])
        s0=np.mean([sat(x[0]) for x in f.values() if x[0] is not None])
        s1=np.mean([sat(x[1]) for x in f.values()])
        dr=np.mean([np.abs(x[1]-x[0]).mean() for x in f.values() if x[0] is not None])
        lines.append(f"| {label(v)} | {len(f)} | {'--' if np.isnan(z) else f'{z:.2f}'} | {s0:.1f} -> {s1:.1f} | {dr:.3f} |")
    txt="\n".join(lines)+"\n"
    (root/"npa_naturalness.md").write_text(txt)
    print(txt)
    folds=sorted({k for f in by.values() for k in f})
    T,G=96,6
    rows=[]
    for v in order:
        tiles=[]
        for k in folds:
            ini,fin,_=by[v].get(k,(None,None,0))
            pair=[]
            for p in (ini,fin):
                if p is None:
                    pair.append(np.full((T,T,3),255,np.uint8))
                else:
                    pair.append(cv2.resize((p.transpose(1,2,0)*255).astype(np.uint8),(T,T),interpolation=cv2.INTER_AREA))
            tiles.append(np.concatenate([pair[0],np.full((T,2,3),255,np.uint8),pair[1]],1))
            tiles.append(np.full((T,G,3),255,np.uint8))
        row=np.concatenate(tiles,1)
        head=np.full((18,row.shape[1],3),255,np.uint8)
        cv2.putText(head,label(v)+"   (each fold: start | final)",(4,13),0,0.42,(0,0,0),1)
        rows+=[head,row,np.full((G,row.shape[1],3),255,np.uint8)]
    sheet=np.concatenate(rows,0)
    cv2.imwrite(str(root/"npa_naturalness_sheet.png"),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
    print(f"written {root/'npa_naturalness.md'} and {root/'npa_naturalness_sheet.png'}")

if __name__=="__main__":
    main()
