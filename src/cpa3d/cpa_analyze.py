"""Pool cpa_eval.py results over folds: ASR per detector and condition, margin over the best control with
photo-level CIs, paired sign-flip tests against the baseline (Holm per detector). Thresholds 0.5 and 0.25.
    python cpa_analyze.py --runs cpa3d_runs"""

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
CONDS=["none","light","shadow","rain","mud"]

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
    ap.add_argument("--eval-dir",default="eval")
    ap.add_argument("--baseline",default="cpa_none")
    ap.add_argument("--report-name",default="cpa3d_report")
    ap.add_argument("--n-boot",type=int,default=4000)
    a=ap.parse_args()
    root=Path(a.runs)
    rows=defaultdict(list) # detector -> rows (all folds)
    folds=defaultdict(set)
    for f in sorted(root.glob(f"fold*/{a.eval_dir}/*/per_instance.json")):
        det=f.parent.name
        rows[det].extend(json.load(open(f)))
        folds[det].add(f.parts[-4])
    if not rows:
        raise SystemExit(f"no {root}/fold*/{a.eval_dir}/*/per_instance.json found")
    dets=sorted(rows,key= lambda d:(d!="surrogate",d))
    r0=rows[dets[0]][0]
    variants=sorted({k.split("_",2)[2] for k in r0 if k.startswith("after_none_")})
    methods=[v for v in variants if v not in CONTROLS]
    order=[a.baseline,"cpa_none","cpa_light","cpa_shadow","cpa_ls","npa","npa_init","pixel"]
    methods=sorted(methods,key= lambda m:(order.index(m) if m in order else len(order),m))
    conds=[c for c in CONDS if f"after_{c}_{variants[0]}" in r0]
    lines,js=[],{}
    P=lines.append
    P(f"# 3-D patch attack results (pooled over folds)\n\nDetectors: "
      +", ".join(f"{d} ({len(folds[d])} folds)" for d in dets)+"\n")
    for thr in (0.5,0.25):
        js[thr]={}
        stats={}
        for d in dets:
            det=[r for r in rows[d] if r["before_conf"]>=thr]
            if not det:
                continue
            groups=[r["group"] for r in det]
            stats[d]={"n":len(det),"groups":groups,"n_photos":len(set(groups)),"rows":det}
            for c in conds:
                series={v:np.array([r[f"after_{c}_{v}"]<thr for r in det],float) for v in variants}
                for m in methods:
                    s={"adversarial":series[m],**{k:series[k] for k in CONTROLS}}
                    stats[d][(c,m)]=cluster_bootstrap(groups,s,n_boot=a.n_boot)
                for k in CONTROLS:
                    stats[d][(c,k)]=float(series[k].mean())
        P(f"\n## Threshold {thr}\n")
        P("### A. ASR, no corruption [95% photo-level CI]\n")
        P("| texture | "+" | ".join(f"{d} (n={stats[d]['n']}, {stats[d]['n_photos']} photos)" for d in stats)+" |")
        P("|---|"+"---|"*len(stats))
        for m in methods:
            cells=[]
            for d in stats:
                b=stats[d][("none",m)]
                lo,hi=b["ci95"]["adversarial"]
                cells.append(f"{b['rate']['adversarial']:.3f} [{lo:.2f}, {hi:.2f}]")
            P(f"| {m} | "+" | ".join(cells)+" |")
        for k in CONTROLS:
            P(f"| *{k} control* | "+" | ".join(f"{stats[d][('none', k)]:.3f}" for d in stats)+" |")
        P("\n### B. Margin over the best control, no corruption [95% CI]\n")
        P("| texture | "+" | ".join(stats)+" |")
        P("|---|"+"---|"*len(stats))
        for m in methods:
            cells=[]
            for d in stats:
                b=stats[d][("none",m)]
                lo,hi=b["margin_ci95"]
                cells.append(f"{b['margin_over_best_control']:+.3f} [{lo:+.2f}, {hi:+.2f}]")
            P(f"| {m} | "+" | ".join(cells)+" |")
        P("\n### C. ASR per test condition\n")
        for d in stats:
            P(f"\n**{d}**\n")
            P("| texture | "+" | ".join(conds)+" | corr. avg |")
            P("|---|"+"---|"*(len(conds)+1))
            for v in methods+list(CONTROLS):
                vals=[stats[d][(c,v)]["rate"]["adversarial"] if v in methods else stats[d][(c,v)] for c in conds]
                avg=np.mean(vals[1:]) if len(vals)>1 else float("nan")
                P(f"| {v} | "+" | ".join(f"{x:.3f}" for x in vals)+f" | {avg:.3f} |")
        if a.baseline in methods and len(methods)>1:
            P(f"\n### D. Paired tests vs `{a.baseline}` (difference in ASR; photos sign-flipped; Holm per detector)\n")
            P("| detector | texture | condition | diff [95% CI] | photos that differ | p | Holm p |")
            P("|---|---|---|---|---|---|---|")
            for d in stats:
                det,groups=stats[d]["rows"],stats[d]["groups"]
                tests=[]
                for m in methods:
                    if m==a.baseline:
                        continue
                    for label,cs in (("none",["none"]),("corr. avg",conds[1:])):
                        if not cs:
                            continue
                        ga=np.mean([[r[f"after_{c}_{a.baseline}"]<thr for c in cs] for r in det],1)
                        gb=np.mean([[r[f"after_{c}_{m}"]<thr for c in cs] for r in det],1)
                        tests.append((m,label,paired_photo_test(groups,ga,gb,n_boot=a.n_boot)))
                hp=holm([t[2]["p"] for t in tests]) if tests else []
                for (m,label,t),h in zip(tests,hp):
                    lo,hi=t["ci95"]
                    P(f"| {d} | {m} | {label} | {t['diff']:+.3f} [{lo:+.2f}, {hi:+.2f}] | "
 f"{t['n_photos_differ']} (min p {t['min_possible_p']:.3g}) | {t['p']:.3g} | {h:.3g} |")
        js[thr]={d:{f"{c}/{v}":(stats[d][(c,v)]["rate"]["adversarial"] if v in methods else stats[d][(c,v)])
                       for c in conds for v in variants}|{"n":stats[d]["n"]} for d in stats}
    out=root/f"{a.report_name}.md"
    out.write_text("\n".join(lines)+"\n")
    json.dump({str(k):v for k,v in js.items()},open(root/f"{a.report_name}.json","w"),indent=2)
    print("\n".join(lines))
    print(f"\nwritten {out}")

if __name__=="__main__":
    main()
