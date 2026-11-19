"""Does the CPA gain appear with paper-like low diversity (one fixed view and heading per tank)?
Gain (ls-none) at low and full diversity and their interaction, seed-averaged, photo-level tests.
    python cpa_diversity.py --runs cpa3d_runs"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import common
from photo_stats import cluster_bootstrap,paired_photo_test

CONTROLS=("gray","noise")
LEVELS={"low":(["cpa_none_fix","cpa_none_fix_s1"],["cpa_ls_fix","cpa_ls_fix_s1"]),
          "full":(["cpa_none","cpa_none_s1"],["cpa_ls","cpa_ls_s1"])}
DET_ORDER=["yolov8s","yolov8m","yolo11s","rtdetr"]

def holm(ps):
    order=np.argsort(ps)
    out,run=[0.0]*len(ps),0.0
    for r,i in enumerate(order):
        run=max(run,min(1.0,(len(ps)-r)*ps[i]))
        out[i]=run
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs",default="cpa3d_runs")
    ap.add_argument("--eval-dir",default="eval_diversity")
    ap.add_argument("--n-boot",type=int,default=4000)
    a=ap.parse_args()
    root=Path(a.runs)
    rows=defaultdict(list)
    for f in sorted(root.glob(f"fold*/{a.eval_dir}/*/per_instance.json")):
        d=f.parent.name
        rows["yolov8s" if d=="surrogate" else d].extend(json.load(open(f)))
    if not rows:
        raise SystemExit(f"no {root}/fold*/{a.eval_dir}/*/per_instance.json")
    dets=[d for d in DET_ORDER if d in rows]
    def g(det,v,thr,conds=("none",)):
        return np.mean([[r[f"after_{c}_{v}"]<thr for c in conds] for r in det],1)
    L=[]
    P=L.append
    P("# CPA gain at low (fixed-view) and full training diversity, 2 seeds averaged\n")
    for thr in (0.5,0.25):
        P(f"\n## Threshold {thr}\n")
        D={t:[r for r in rows[t] if r["before_conf"]>=thr] for t in dets}
        P("### A. ASR (margin over best control [95% photo CI]), no corruption\n")
        P("| texture (2 seeds avg) | "+" | ".join(f"{t} (n={len(D[t])})" for t in dets)+" |")
        P("|---|"+"---|"*len(dets))
        for lev,(vn,vl) in LEVELS.items():
            for name,vs in (("cpa_none",vn),("cpa_ls",vl)):
                cells=[]
                for t in dets:
                    det=D[t]
                    if not det:
                        cells.append("--")
                        continue
                    groups=[r["group"] for r in det]
                    s={"adversarial":np.mean([g(det,v,thr) for v in vs],0),
                         **{k:g(det,k,thr) for k in CONTROLS}}
                    b=cluster_bootstrap(groups,s,n_boot=a.n_boot)
                    lo,hi=b["margin_ci95"]
                    cells.append(f"{b['rate']['adversarial']:.3f} ({b['margin_over_best_control']:+.3f} [{lo:+.2f}, {hi:+.2f}])")
                P(f"| {name}, {lev} diversity | "+" | ".join(cells)+" |")
        for cl,cs in (("no corruption",("none",)),("corruption avg",("light","shadow","rain","mud"))):
            P(f"\n### B/C. CPA gain (ls - none) and interaction -- test condition: {cl}\n")
            P("| tested on | gain, low diversity | gain, full diversity | interaction low - full | photos | p | Holm p |")
            P("|---|---|---|---|---|---|---|")
            res=[]
            pooled={"g":[],"lo":[],"fu":[]}
            for t in dets:
                det=D[t]
                if not det or f"after_{cs[0]}_cpa_ls_fix" not in det[0]:
                    continue
                groups=[r["group"] for r in det]
                gain={}
                for lev,(vn,vl) in LEVELS.items():
                    gain[lev]=(np.mean([g(det,v,thr,cs) for v in vl],0)-np.mean([g(det,v,thr,cs) for v in vn],0))
                tl=paired_photo_test(groups,np.zeros(len(det)),gain["low"],n_boot=a.n_boot)
                tf=paired_photo_test(groups,np.zeros(len(det)),gain["full"],n_boot=a.n_boot)
                ti=paired_photo_test(groups,gain["full"],gain["low"],n_boot=a.n_boot)
                res.append((t,tl,tf,ti))
                if t!="yolov8s":
                    pooled["g"]+=groups;pooled["lo"]+=list(gain["low"]);pooled["fu"]+=list(gain["full"])
            if pooled["g"]:
                z=np.zeros(len(pooled["g"]))
                res.append(("BLACK-BOX pooled",paired_photo_test(pooled["g"],z,np.array(pooled["lo"]),n_boot=a.n_boot),
                            paired_photo_test(pooled["g"],z,np.array(pooled["fu"]),n_boot=a.n_boot),
                            paired_photo_test(pooled["g"],np.array(pooled["fu"]),np.array(pooled["lo"]),n_boot=a.n_boot)))
            hp=holm([x[3]["p"] for x in res])
            for (t,tl,tf,ti),h in zip(res,hp):
                f= lambda r:f"{r['diff']:+.3f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]"
                P(f"| {t} | {f(tl)} | {f(tf)} | **{f(ti)}** | {ti['n_photos_differ']} | {ti['p']:.3g} | {h:.3g} |")
    out=root/"diversity_report.md"
    out.write_text("\n".join(L)+"\n")
    print("\n".join(L))
    print(f"\nwritten {out}")

if __name__=="__main__":
    main()
