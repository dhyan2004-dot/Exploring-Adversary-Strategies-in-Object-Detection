"""Pool cv_run.py results over folds: detector metrics, attack success vs gray/noise controls (margin),
photo-level bootstrap CIs, paired sign-flip tests with Holm correction, seed and fold consistency.
    python cv_analyze.py --out cv_runs"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from photo_stats import cluster_bootstrap,paired_photo_test

CONDITIONS=["none","light","shadow","rain","mud","mixed"]
SINGLES=["light","shadow","rain","mud"]
TRAIN_KINDS={"eot":[],"cpa_light":["light"],"cpa_shadow":["shadow"],
               "cpa_ls":["light","shadow"],"cpa_all":SINGLES}
VARIANTS=["adversarial","gray","noise"]

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="cv_runs")
    p.add_argument("--baseline",default="eot",help="Method every other method is compared against.")
    p.add_argument("--n-boot",type=int,default=4000)
    return p.parse_args()

class Report:
    def __init__(self):
        self.lines=[]
    def __call__(self,s=""):
        print(s)
        self.lines.append(s)

def ms(xs):
    xs=np.asarray(xs,dtype=float)
    return f"{xs.mean():.3f} ± {xs.std(ddof=1) if len(xs) > 1 else 0.0:.3f}"

def fmt_p(p):
    return "<0.0001" if p<1e-4 else f"{p:.4f}"

def holm(pvals):
    order=np.argsort(pvals)
    m=len(pvals)
    adj=np.empty(m)
    running=0.0
    for rank,i in enumerate(order):
        running=max(running,(m-rank)*pvals[i])
        adj[i]=min(1.0,running)
    return adj

def detector_section(out:Path,rep:Report,js):
    folds=sorted(out.glob("fold*/detector_metrics"))
    if not folds:
        return
    rep(f"## 1. Detector ({len(folds)}-fold, photo-grouped)\n")
    res={}
    for split,label in (("val","held-out photos (never trained on; also used to pick the checkpoint -> slightly optimistic)"),
                         ("test","official Testing split (independent of all folds)")):
        rows=[]
        for d in folds:
            f=d/f"{split}_metrics.json"
            if f.exists():
                rows.append((int(d.parent.name[4:]),json.load(open(f))))
        if not rows:
            continue
        classes=[]
        for _,r in rows:
            for c in r["per_class"]:
                if c["name"] not in classes:
                    classes.append(c["name"])
        def ap(r,name):
            hit=[c["AP50"] for c in r["per_class"] if c["name"]==name]
            return hit[0] if hit else None # class absent from this fold's held-out photos
        def cell_ms(vals):
            vals=[v for v in vals if v is not None]
            return ms(vals)+(f" (n={len(vals)})" if len(vals)<len(rows) else "") if vals else "–"
        rep(f"**{label}** — {len(rows)} folds")
        rep("| fold | mAP50 | mAP50-95 | "+" | ".join(c+" AP50" for c in classes)+" |")
        rep("|---|---|---|"+"---|"*len(classes))
        for fo,r in rows:
            rep(f"| {fo} | {r['overall']['mAP50']:.3f} | {r['overall']['mAP50-95']:.3f} | "
                +" | ".join("–" if ap(r,c) is None else f"{ap(r, c):.3f}" for c in classes)+" |")
        rep(f"| **mean ± sd** | {ms([r['overall']['mAP50'] for _, r in rows])} | "
 f"{ms([r['overall']['mAP50-95'] for _, r in rows])} | "
            +" | ".join(cell_ms([ap(r,c) for _,r in rows]) for c in classes)+" |")
        rep("")
        res[split]={"mAP50":[r["overall"]["mAP50"] for _,r in rows],
                      "per_class":{c:[ap(r,c) for _,r in rows] for c in classes}}
    js["detector"]=res

def load_patch_results(out:Path):
    """-> runs[(method, seed)][fold] = (rows, thr)"""
    runs=defaultdict(dict)
    for pi in sorted(out.glob("fold*/eval/*/per_instance.json")):
        fold=int(pi.parts[-4][4:])
        method,seed=pi.parts[-2].rsplit("_s",1)
        rows=json.load(open(pi))
        thr=json.load(open(pi.parent/"attack_summary.json"))["conf_threshold"]
        runs[(method,int(seed))][fold]=(rows,thr)
    return runs

def build_pool(runs,rep):
    """Align all runs on identical (fold, instance) rows; keep detected-before-patch instances only."""
    keys=sorted(runs)
    folds=sorted(set.intersection(*[set(runs[k]) for k in keys]))
    missing=sorted(set.union(*[set(runs[k]) for k in keys])-set(folds))
    if missing:
        rep(f"_Folds {missing} are incomplete for at least one method/seed and are excluded from the pooled analysis._\n")
    thr=runs[keys[0]][folds[0]][1]
    ref_key=keys[0]
    groups,fold_of,sel=[],[],{k:[] for k in keys}
    for f in folds:
        ref_rows=runs[ref_key][f][0]
        for k in keys:
            rows=runs[k][f][0]
            assert len(rows)==len(ref_rows) and all(a["image"]==b["image"] and a["instance"]==b["instance"]
                                                      for a,b in zip(rows,ref_rows)), \
                f"fold {f}: {k} was evaluated on different instances than {ref_key}; rerun with identical settings."
        det=[i for i,r in enumerate(ref_rows) if r["before_conf"]>=thr]
        for i in det:
            groups.append(ref_rows[i]["group"])
            fold_of.append(f)
        for k in keys:
            rows=runs[k][f][0]
            sel[k]+=[rows[i] for i in det]
    return folds,thr,np.array(groups),np.array(fold_of),sel

def gone_arrays(sel,thr):
    """per (method, seed): cond -> variant -> 0/1 array (1 = the object disappeared)"""
    out={}
    for k,rows in sel.items():
        out[k]={c:{v:np.array([r[f"after_{c}_{v}"]<thr for r in rows],dtype=float) for v in VARIANTS}
                  for c in CONDITIONS if rows and f"after_{c}_adversarial" in rows[0]}
    return out

def per_method(gone,methods_seeds):
    """Average over patch seeds -> method -> cond -> variant -> float array in [0,1]."""
    res={}
    for m,seeds in methods_seeds.items():
        res[m]={c:{v:np.mean([gone[(m,s)][c][v] for s in seeds],axis=0) for v in VARIANTS}
                  for c in gone[(m,seeds[0])]}
    return res

def group_series(pm,conds):
    return {v:np.mean([pm[c][v] for c in conds],axis=0) for v in VARIANTS}

def patch_section(out,rep,js,baseline,n_boot):
    runs=load_patch_results(out)
    if not runs:
        rep("_No patch evaluation results found yet._")
        return
    methods_seeds=defaultdict(list)
    for m,s in sorted(runs):
        methods_seeds[m].append(s)
    methods=[m for m in TRAIN_KINDS if m in methods_seeds]+[m for m in methods_seeds if m not in TRAIN_KINDS]
    folds,thr,groups,fold_of,sel=build_pool(runs,rep)
    gone=gone_arrays(sel,thr)
    pm=per_method(gone,{m:methods_seeds[m] for m in methods})
    n_ph,n_inst=len(set(groups)),len(groups)
    if n_inst==0:
        rep("## 2. Patches\n\n_The detectors found no target instances before patching (conf threshold "
 f"{thr}), so there is nothing to attack. Expected in the 1-epoch smoke test; in a real run it "
 "means the detector is broken (check imgsz / weights)._\n")
        return
    tot_ph={f:len(set(groups[fold_of==f])) for f in folds}
    rep(f"## 2. Patches (pooled over folds {folds})\n")
    rep(f"Target instances detected before patching: **{n_inst}** from **{n_ph} distinct photos** "
 f"(per fold: {tot_ph}). Attack success = a detected object drops below conf {thr}. "
 f"Seeds per method: { {m: len(methods_seeds[m]) for m in methods} }. "
 f"All intervals resample whole photos.\n")
    js["patch"]={"n_instances":n_inst,"n_photos":n_ph,"folds":folds,"groups":{}}
    rep("Headline columns weight every instance equally. The last column weights every PHOTO equally; "
 "if the two disagree, a few crowded photos are driving the result.\n")
    cond_groups={"none (no corruption)":["none"],"light+shadow+rain+mud (avg)":SINGLES,"mixed (2 stacked)":["mixed"]}
    rep("### 2a. Attack success vs. the best control (gray / noise square of the same size and place)\n")
    rep("| method | condition | patch hides object | best control | **margin** | margin 95% CI (photo-resampled) "
 "| photo-weighted margin [95% CI] |")
    rep("|---|---|---|---|---|---|---|")
    for m in methods:
        for gname,conds in cond_groups.items():
            conds=[c for c in conds if c in pm[m]]
            if not conds:
                continue
            ser=group_series(pm[m],conds)
            r=cluster_bootstrap(groups,ser,n_boot=n_boot)
            ctrl=max(r["rate"]["gray"],r["rate"]["noise"])
            lo,hi=r["ci95"]["adversarial"]
            mlo,mhi=r["margin_ci95"]
            flag="" if mlo>0 else "  (CI includes 0)"
            rp=cluster_bootstrap(groups,ser,n_boot=n_boot,weighting="photo")
            plo,phi=rp["margin_ci95"]
            rep(f"| {m} | {gname} | {r['rate']['adversarial']:.3f} [{lo:.2f}, {hi:.2f}] | {ctrl:.3f} | "
 f"**{r['margin_over_best_control']:+.3f}** | [{mlo:+.3f}, {mhi:+.3f}]{flag} | "
 f"{rp['margin_over_best_control']:+.3f} [{plo:+.3f}, {phi:+.3f}] |")
            js["patch"]["groups"][f"{m}|{gname}"]={"instance_weighted":r,"photo_weighted":rp}
    rep("")
    rep("### 2b. Every condition (patch hides object / margin over best control)\n")
    rep("| method | "+" | ".join(CONDITIONS)+" |")
    rep("|---|"+"---|"*len(CONDITIONS))
    for m in methods:
        cells=[]
        for c in CONDITIONS:
            if c not in pm[m]:
                cells.append("–")
                continue
            a=pm[m][c]["adversarial"].mean()
            ctrl=max(pm[m][c]["gray"].mean(),pm[m][c]["noise"].mean())
            tag="*" if c in TRAIN_KINDS.get(m,[]) else ""
            cells.append(f"{a:.2f} / {a - ctrl:+.2f}{tag}")
        rep(f"| {m} | "+" | ".join(cells)+" |")
    rep("\n`*` = this corruption was applied to the patch during that method's training; unmarked corruption columns were unseen.\n")
    rep("### 2c. Paired comparison between methods (same instances, same corruption draws)\n")
    rep("Positive difference = the second method hides MORE objects. p = photo-level sign-flip test, "
 "Holm-adjusted across this table.\n")
    pairs=[]
    for m in methods:
        if m!=baseline and baseline in pm:
            pairs.append((baseline,m))
    if "cpa_ls" in pm:
        for single in ("cpa_light","cpa_shadow"):
            if single in pm:
                pairs.append((single,"cpa_ls"))
    comps=[]
    for a,b in pairs:
        for gname,conds in cond_groups.items():
            conds=[c for c in conds if c in pm[a] and c in pm[b]]
            if not conds:
                continue
            ga=group_series(pm[a],conds)["adversarial"]
            gb=group_series(pm[b],conds)["adversarial"]
            t=paired_photo_test(groups,ga,gb,n_boot=n_boot)
            comps.append((a,b,gname,t))
    if comps:
        adj=holm(np.array([c[3]["p"] for c in comps]))
        rep("| A | B | condition | A | B | B − A | 95% CI | photos that differ | smallest possible p | p | p (Holm) |")
        rep("|---|---|---|---|---|---|---|---|---|---|---|")
        for (a,b,gname,t),pa in zip(comps,adj):
            rep(f"| {a} | {b} | {gname} | {t['rate_a']:.3f} | {t['rate_b']:.3f} | {t['diff']:+.3f} | "
 f"[{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] | {t['n_photos_differ']} | {t['min_possible_p']:.3f} | "
 f"{fmt_p(t['p'])} | {fmt_p(pa)} |")
        js["patch"]["paired"]=[{"a":a,"b":b,"cond":g,**t,"p_holm":float(pa)}
                                 for (a,b,g,t),pa in zip(comps,adj)]
        rep("\nOnly photos where the two methods differ carry information, so the p-value cannot go below "
 "'smallest possible p'. With few photos the bootstrap CI is too narrow; when CI and p disagree, "
 "quote the (Holm-adjusted) p.\n")
    rep("### 2d. Consistency across folds (margin over best control, no corruption)\n")
    rep("| method | "+" | ".join(f"fold {f}" for f in folds)+" | mean ± sd |")
    rep("|---|"+"---|"*(len(folds)+1))
    for m in methods:
        vals=[]
        for f in folds:
            mk=fold_of==f
            a=pm[m]["none"]["adversarial"][mk].mean()
            ctrl=max(pm[m]["none"]["gray"][mk].mean(),pm[m]["none"]["noise"][mk].mean())
            vals.append(a-ctrl)
        rep(f"| {m} | "+" | ".join(f"{v:+.2f}" for v in vals)+f" | {ms(vals)} |")
    rep("")
    if any(len(v)>1 for v in methods_seeds.values()):
        rep("### 2e. Seed variance (margin over best control, pooled over all photos; one value per patch seed)\n")
        rep("| method | none: per seed | none: mean ± sd | corrupted avg: mean ± sd |")
        rep("|---|---|---|---|")
        for m in methods:
            per_none,per_cor=[],[]
            for s in methods_seeds[m]:
                g=gone[(m,s)]
                a=g["none"]["adversarial"].mean()
                per_none.append(a-max(g["none"]["gray"].mean(),g["none"]["noise"].mean()))
                ga=np.mean([g[c]["adversarial"].mean() for c in SINGLES])
                gc=max(np.mean([g[c]["gray"].mean() for c in SINGLES]),np.mean([g[c]["noise"].mean() for c in SINGLES]))
                per_cor.append(ga-gc)
            rep(f"| {m} | {', '.join(f'{v:+.2f}' for v in per_none)} | {ms(per_none)} | {ms(per_cor)} |")
        rep("")

def mapdrop_section(out,rep,js):
    files=sorted(out.glob("fold*/mapdrop/*/mapdrop_summary.json"))
    if not files:
        return
    per=defaultdict(lambda:defaultdict(list)) # method -> variant -> [(fold, mAP drop, class drop)]
    clean=defaultdict(list)
    target=None
    for f in files:
        fold=int(f.parts[-4][4:])
        method=f.parts[-2].rsplit("_s",1)[0]
        d=json.load(open(f))
        target=d["target_class"]
        clean[(method,fold)].append((d["results"]["clean"]["mAP50"],d["results"]["clean"]["AP50_by_class"].get(target,np.nan)))
        for v,dr in d["drops"].items():
            per[method][v].append((fold,dr["mAP50_drop"],dr[f"{target}_AP50_drop"]))
    rep(f"## 3. Full-image mAP50 drop with the patch on every {target} instance (held-out photos)\n")
    rep("Mean over folds ± sd (seeds averaged inside a fold). The adversarial drop only counts beyond the gray / noise drops.\n")
    rep(f"| method | clean {target} AP50 | adversarial: {target} AP50 drop | gray | noise | adversarial: overall mAP50 drop | gray | noise |")
    rep("|---|---|---|---|---|---|---|---|")
    js["mapdrop"]={}
    for m,vs in per.items():
        def by_fold(v,idx):
            acc=defaultdict(list)
            for fo,*vals in vs[v]:
                acc[fo].append(vals[idx])
            return [np.mean(x) for _,x in sorted(acc.items())]
        cl=[np.mean([c[1] for c in clean[(m,fo)]]) for fo in sorted({k[1] for k in clean if k[0]==m})]
        row={v:(by_fold(v,1),by_fold(v,0)) for v in ("adversarial","gray","noise")}
        rep(f"| {m} | {ms(cl)} | "+" | ".join(ms(row[v][0]) for v in ("adversarial","gray","noise"))+" | "
            +" | ".join(ms(row[v][1]) for v in ("adversarial","gray","noise"))+" |")
        js["mapdrop"][m]={v:{"class_ap50_drop":row[v][0],"map50_drop":row[v][1]} for v in row}
    rep("")

def main():
    args=parse_args()
    out=Path(args.out)
    rep,js=Report(),{}
    rep("# Cross-validated results (photo = unit of analysis)\n")
    detector_section(out,rep,js)
    patch_section(out,rep,js,args.baseline,args.n_boot)
    mapdrop_section(out,rep,js)
    (out/"cv_report.md").write_text("\n".join(rep.lines))
    (out/"cv_report.json").write_text(json.dumps(js,indent=1,default=float))
    print(f"\nWrote {out / 'cv_report.md'} and {out / 'cv_report.json'}")

if __name__=="__main__":
    main()
