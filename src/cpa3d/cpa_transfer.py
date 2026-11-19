"""Transfer matrix: every attack trained on each of the 4 detectors and tested on all 4 (diagonal=white-box,
off-diagonal=black-box). Cells: ASR and margin over the best control; paired tests of cpa_ls-cpa_none
and other attacks vs cpa_none over the black-box pairs (Holm).
    python cpa_transfer.py --runs cpa3d_runs"""

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
    ap.add_argument("--eval-dir",default="eval_transfer")
    ap.add_argument("--n-boot",type=int,default=4000)
    ap.add_argument("--report-name",default="transfer_report")
    a=ap.parse_args()
    root=Path(a.runs)
    rows=defaultdict(list)
    for f in sorted(root.glob(f"fold*/{a.eval_dir}/*/per_instance.json")):
        d=f.parent.name
        rows["yolov8s" if d=="surrogate" else d].extend(json.load(open(f)))
    if not rows:
        raise SystemExit(f"no {root}/fold*/{a.eval_dir}/*/per_instance.json")
    dets=[d for d in DET_ORDER if d in rows]+[d for d in rows if d not in DET_ORDER]
    r0=rows[dets[0]][0]
    variants=[k[len("after_none_"):] for k in r0 if k.startswith("after_none_") and "@" in k]
    pref=["cpa_none","cpa_ls","npa","npa_stop"]
    attacks=sorted({v.split("@")[0] for v in variants},key= lambda x:(pref.index(x) if x in pref else 9,x))
    surrs=[d for d in DET_ORDER if any(v.endswith("@"+d) for v in variants)]
    def gone(det,v,thr):
        return np.array([r[f"after_none_{v}"]<thr for r in det],float)
    L=[]
    P=L.append
    P("# Transfer matrix (pooled over folds, no corruption at test)\n")
    P("Rows: detector the patch was TRAINED on. Columns: detector it is TESTED on. "
 "**Bold** = white-box (same detector). Cell: ASR (margin over best control [95% photo CI]).\n")
    for thr in (0.5,0.25):
        P(f"\n## Threshold {thr}\n")
        D={t:[r for r in rows[t] if r["before_conf"]>=thr] for t in dets}
        P("| tested on | "+" | ".join(f"{t} (n={len(D[t])}, {len({r['group'] for r in D[t]})} photos)" for t in dets)+" |")
        P("|---|"+"---|"*len(dets))
        P("| *best control* | "+" | ".join(
            f"{max(gone(D[t], k, thr).mean() for k in CONTROLS):.3f}" if D[t] else "--" for t in dets)+" |")
        summary={}
        for at in attacks:
            P(f"| **{at}** | "+" | ".join("" for _ in dets)+" |")
            wb,bb=[],[]
            for s in surrs:
                v=f"{at}@{s}"
                cells=[]
                for t in dets:
                    det=D[t]
                    if not det or f"after_none_{v}" not in det[0]:
                        cells.append("--")
                        continue
                    groups=[r["group"] for r in det]
                    b=cluster_bootstrap(groups,{"adversarial":gone(det,v,thr),
                                                   **{k:gone(det,k,thr) for k in CONTROLS}},n_boot=a.n_boot)
                    lo,hi=b["margin_ci95"]
                    txt=f"{b['rate']['adversarial']:.3f} ({b['margin_over_best_control']:+.3f} [{lo:+.2f}, {hi:+.2f}])"
                    cells.append(f"**{txt}**" if s==t else txt)
                    (wb if s==t else bb).append(b["margin_over_best_control"])
                P(f"| &nbsp;&nbsp;trained on {s} | "+" | ".join(cells)+" |")
            summary[at]=(np.mean(wb) if wb else float("nan"),np.mean(bb) if bb else float("nan"),len(bb))
        P("\n**Average margin over control** (white-box = diagonal, black-box = all off-diagonal pairs)\n")
        P("| attack | white-box | black-box | black-box / white-box |")
        P("|---|---|---|---|")
        for at,(w,b,n) in summary.items():
            ratio=f"{b / w:.2f}" if w>0.01 else "--"
            P(f"| {at} | {w:+.3f} | {b:+.3f} ({n} pairs) | {ratio} |" if not np.isnan(w) else f"| {at} | -- | -- | -- |")
        comps=[("cpa_none",b,"CPA claim: cpa_ls - cpa_none" if b=="cpa_ls" else f"{b} - cpa_none")
                 for b in attacks if b!="cpa_none"]
        for A,B,title in comps:
            if A not in attacks or B not in attacks:
                continue
            P(f"\n### {title} (positive = {B} transfers better)\n")
            P("| tested on | black-box surrogates | diff [95% CI] | photos that differ | p | Holm p |")
            P("|---|---|---|---|---|---|")
            tests=[]
            pooled_g,pooled_a,pooled_b=[],[],[]
            for t in dets:
                det=D[t]
                bbs=[s for s in surrs if s!=t and det and f"after_none_{A}@{s}" in det[0]
                       and f"after_none_{B}@{s}" in det[0]]
                if not bbs:
                    continue
                ga=np.mean([gone(det,f"{A}@{s}",thr) for s in bbs],0)
                gb=np.mean([gone(det,f"{B}@{s}",thr) for s in bbs],0)
                groups=[r["group"] for r in det]
                tests.append((t,", ".join(bbs),paired_photo_test(groups,ga,gb,n_boot=a.n_boot)))
                pooled_g+=groups;pooled_a+=list(ga);pooled_b+=list(gb)
            if pooled_g:
                tests.append(("ALL (pooled)","all off-diagonal",
                              paired_photo_test(pooled_g,np.array(pooled_a),np.array(pooled_b),n_boot=a.n_boot)))
            for (t,s,r),h in zip(tests,holm([x[2]["p"] for x in tests])):
                lo,hi=r["ci95"]
                P(f"| {t} | {s} | {r['diff']:+.3f} [{lo:+.2f}, {hi:+.2f}] | {r['n_photos_differ']} | {r['p']:.3g} | {h:.3g} |")
    out=root/f"{a.report_name}.md"
    out.write_text("\n".join(L)+"\n")
    print("\n".join(L))
    print(f"\nwritten {out}")

if __name__=="__main__":
    main()
