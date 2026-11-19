"""Ablation tables for the CPA gain (light+shadow minus none): patch size 1/1.5/2 m, two seeds,
corruption probability 1.0 vs 0.5.
    python cpa_ablation.py --runs cpa3d_runs"""

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
CORR=["light","shadow","rain","mud"]
SIZES=[("1.0 m","eval_abl_1m","cpa_none","cpa_ls"),
         ("1.5 m","eval_abl_1p5","cpa_none_1p5","cpa_ls_1p5"),
         ("2.0 m","eval_abl_2m","cpa_none_2m","cpa_ls_2m")]

def holm(ps):
    order=np.argsort(ps)
    out,run=[0.0]*len(ps),0.0
    for r,i in enumerate(order):
        run=max(run,min(1.0,(len(ps)-r)*ps[i]))
        out[i]=run
    return out

def load(root,eval_dir):
    rows=defaultdict(list)
    for f in sorted(root.glob(f"fold*/{eval_dir}/*/per_instance.json")):
        rows[f.parent.name].extend(json.load(open(f)))
    return rows

def gone(det,v,thr,conds):
    return np.mean([[r[f"after_{c}_{v}"]<thr for c in conds] for r in det],1)

def margin(det,v,thr,n_boot):
    groups=[r["group"] for r in det]
    s={"adversarial":gone(det,v,thr,["none"]),**{k:gone(det,k,thr,["none"]) for k in CONTROLS}}
    return cluster_bootstrap(groups,s,n_boot=n_boot)

def fmt_m(b):
    lo,hi=b["margin_ci95"]
    return f"{b['rate']['adversarial']:.3f} ({b['margin_over_best_control']:+.3f} [{lo:+.2f}, {hi:+.2f}])"

def test_rows(P,d,det,pairs,thr,n_boot,conds_sets):
    """pairs: [(label, a_list, b_list)] -- a/b are lists of variants averaged per instance."""
    groups=[r["group"] for r in det]
    res=[]
    for label,A,B in pairs:
        for cl,cs in conds_sets:
            ga=np.mean([gone(det,v,thr,cs) for v in A],0)
            gb=np.mean([gone(det,v,thr,cs) for v in B],0)
            res.append((label,cl,paired_photo_test(groups,ga,gb,n_boot=n_boot)))
    hp=holm([t["p"] for _,_,t in res])
    for (label,cl,t),h in zip(res,hp):
        lo,hi=t["ci95"]
        P(f"| {d} | {label} | {cl} | {t['diff']:+.3f} [{lo:+.2f}, {hi:+.2f}] | {t['n_photos_differ']} | "
 f"{t['p']:.3g} | {h:.3g} |")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs",default="cpa3d_runs")
    ap.add_argument("--n-boot",type=int,default=4000)
    a=ap.parse_args()
    root=Path(a.runs)
    data={ed:load(root,ed) for _,ed,_,_ in SIZES}
    if not any(data.values()):
        raise SystemExit(f"no ablation results under {root}/fold*/eval_abl_*")
    dets=sorted({d for v in data.values() for d in v},key= lambda d:(d!="surrogate",d))
    lines=[]
    P=lines.append
    P("# 3-D CPA ablations (pooled over folds)\n")
    P("Cells: ASR (margin over best control [95% CI])\n")
    tests_conds=[("none",["none"]),("corr. avg",CORR)]
    for thr in (0.5,0.25):
        P(f"\n## Threshold {thr}\n")
        P("### 1. Patch size (no corruption at test)\n")
        P("| sheet | texture | "+" | ".join(dets)+" |")
        P("|---|---|"+"---|"*len(dets))
        for size,ed,vn,vl in SIZES:
            for v in (vn,vl):
                cells=[]
                for d in dets:
                    det=[r for r in data[ed].get(d,[]) if r["before_conf"]>=thr]
                    cells.append(fmt_m(margin(det,v,thr,a.n_boot)) if det and f"after_none_{v}" in det[0] else "--")
                P(f"| {size} | {v} | "+" | ".join(cells)+" |")
            cells=[]
            for d in dets:
                det=[r for r in data[ed].get(d,[]) if r["before_conf"]>=thr]
                cells.append(f"{max(gone(det, k, thr, ['none']).mean() for k in CONTROLS):.3f}" if det else "--")
            P(f"| {size} | *best control* | "+" | ".join(cells)+" |")
        P("\n**cpa_ls - cpa_none per size**\n")
        P("| detector | sheet | condition | diff [95% CI] | photos that differ | p | Holm p |")
        P("|---|---|---|---|---|---|---|")
        for d in dets:
            res_pairs=[]
            for size,ed,vn,vl in SIZES:
                det=[r for r in data[ed].get(d,[]) if r["before_conf"]>=thr]
                if det and f"after_none_{vl}" in det[0]:
                    res_pairs.append((size,det,vn,vl))
            allres=[]
            for size,det,vn,vl in res_pairs:
                groups=[r["group"] for r in det]
                for cl,cs in tests_conds:
                    allres.append((size,cl,paired_photo_test(groups,gone(det,vn,thr,cs),gone(det,vl,thr,cs),
                                                               n_boot=a.n_boot)))
            for (size,cl,t),h in zip(allres,holm([t["p"] for _,_,t in allres])):
                lo,hi=t["ci95"]
                P(f"| {d} | {size} | {cl} | {t['diff']:+.3f} [{lo:+.2f}, {hi:+.2f}] | {t['n_photos_differ']} | "
 f"{t['p']:.3g} | {h:.3g} |")
        ed="eval_abl_1m"
        P("\n### 2. Seed variance (1 m, no corruption at test)\n")
        P("| texture | "+" | ".join(dets)+" |")
        P("|---|"+"---|"*len(dets))
        for v in ("cpa_none","cpa_none_s1","cpa_ls","cpa_ls_s1"):
            cells=[]
            for d in dets:
                det=[r for r in data[ed].get(d,[]) if r["before_conf"]>=thr]
                cells.append(fmt_m(margin(det,v,thr,a.n_boot)) if det and f"after_none_{v}" in det[0] else "--")
            P(f"| {v} | "+" | ".join(cells)+" |")
        P("\n| detector | comparison | condition | diff [95% CI] | photos that differ | p | Holm p |")
        P("|---|---|---|---|---|---|---|")
        pairs=[("seed noise: none s1 - s0",["cpa_none"],["cpa_none_s1"]),
                 ("seed noise: ls s1 - s0",["cpa_ls"],["cpa_ls_s1"]),
                 ("ls - none, seed-averaged",["cpa_none","cpa_none_s1"],["cpa_ls","cpa_ls_s1"])]
        for d in dets:
            det=[r for r in data[ed].get(d,[]) if r["before_conf"]>=thr]
            if det and "after_none_cpa_ls_s1" in det[0]:
                test_rows(P,d,det,pairs,thr,a.n_boot,tests_conds)
        P("\n### 3. Corruption strength during training (1 m)\n")
        P("| texture | "+" | ".join(f"{d} none / corr. avg" for d in dets)+" |")
        P("|---|"+"---|"*len(dets))
        for v in ("cpa_none","cpa_ls","cpa_ls_p1"):
            cells=[]
            for d in dets:
                det=[r for r in data[ed].get(d,[]) if r["before_conf"]>=thr]
                ok=det and f"after_none_{v}" in det[0]
                cells.append(f"{gone(det, v, thr, ['none']).mean():.3f} / {gone(det, v, thr, CORR).mean():.3f}" if ok else "--")
            P(f"| {v} (p={'0' if v == 'cpa_none' else ('1.0' if v.endswith('p1') else '0.5')}) | "+" | ".join(cells)+" |")
        P("\n| detector | comparison | condition | diff [95% CI] | photos that differ | p | Holm p |")
        P("|---|---|---|---|---|---|---|")
        pairs=[("ls p=1.0 - none",["cpa_none"],["cpa_ls_p1"]),("ls p=1.0 - ls p=0.5",["cpa_ls"],["cpa_ls_p1"])]
        for d in dets:
            det=[r for r in data[ed].get(d,[]) if r["before_conf"]>=thr]
            if det and "after_none_cpa_ls_p1" in det[0]:
                test_rows(P,d,det,pairs,thr,a.n_boot,tests_conds)
    out=root/"cpa3d_ablation_report.md"
    out.write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
    print(f"\nwritten {out}")

if __name__=="__main__":
    main()
