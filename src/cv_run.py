"""Run the 2-D pipeline per fold (resumable): detector training/evaluation, patch training (EOT and CPA
variants), per-instance evaluation and mAP drop on the held-out photos.
    python cv_run.py --root <yolo_dataset> --out cv_runs --folds all"""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent

METHODS={
    "eot":"",
    "cpa_light":"light",
    "cpa_shadow":"shadow",
    "cpa_ls":"light,shadow",
    "cpa_all":"all",
}

def parse_args():
    p=argparse.ArgumentParser(description="Per-fold detector + patch pipeline.")
    p.add_argument("--root",required=True,help="yolo_dataset folder (the one containing cv/).")
    p.add_argument("--out",default="cv_runs",help="Output folder (outputs are written under <out>/fold{k}/).")
    p.add_argument("--folds",default="all",help="Comma-separated fold numbers, or 'all'.")
    p.add_argument("--stages",default="detector,patch,eval,mapdrop")
    p.add_argument("--methods",default="eot,cpa_light,cpa_shadow,cpa_ls")
    p.add_argument("--seeds",default="0",help="Patch-training seeds, e.g. 0,1,2 (each is a separate patch).")
    p.add_argument("--target-class",default="Tank")
    p.add_argument("--patch-scale",type=float,default=0.4,
                   help="Patch width as a fraction of the object's box width (0.4 ~ 16%% of the box area).")
    p.add_argument("--det-args",default="--model yolov8s.pt --epochs 15 --imgsz 960 --batch 8 --patience 5 --workers 4",
                   help="Extra arguments for train_yolov8.py. SET THIS to your best earlier experiment "
 "(e.g. add --freeze N --lr0 X). Quote the whole string.")
    p.add_argument("--patch-epochs",type=int,default=20)
    p.add_argument("--instances-per-epoch",type=int,default=600)
    p.add_argument("--patch-args",default="",help="Extra arguments appended to patch_train.py.")
    p.add_argument("--max-files-per-photo",type=int,default=6,
                   help="Held-out files per photo used by patch_evaluate / patch_map_drop.")
    p.add_argument("--eval-seed",type=int,default=0,
                   help="Seed for the evaluation corruption draws. Keep it FIXED across methods: this is what "
 "makes the method comparison paired.")
    p.add_argument("--device",default=None,help="Passed to the sub-scripts only if set.")
    p.add_argument("--force",action="store_true")
    p.add_argument("--keep-going",action="store_true",help="Continue with the next job if one fails.")
    p.add_argument("--dry-run",action="store_true",help="Print the commands without running them.")
    return p.parse_args()

def run(cmd,log_path:Path,args,label):
    cmd=[str(c) for c in cmd]
    print(f"\n[{time.strftime('%H:%M:%S')}] {label}\n    "+" ".join(shlex.quote(c) for c in cmd),flush=True)
    if args.dry_run:
        return True
    log_path.parent.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    with open(log_path,"w") as lf:
        rc=subprocess.call(cmd,cwd=HERE,stdout=lf,stderr=subprocess.STDOUT)
    dt=time.time()-t0
    if rc!=0:
        tail="".join(open(log_path).readlines()[-25:])
        print(f"    FAILED (exit {rc}) after {dt:.0f}s. Last lines of {log_path}:\n{tail}",flush=True)
        if not args.keep_going:
            raise SystemExit(1)
        return False
    print(f"    done in {dt:.0f}s  (log: {log_path})",flush=True)
    return True

def stage_ok(out_dir:Path,cfg:dict,done_file:str,args,honor_force=True):
    """True if this stage already finished with the SAME settings."""
    marker=out_dir/"stage_done.json"
    if (honor_force and args.force) or not (out_dir/done_file).exists() or not marker.exists():
        return False
    old=json.load(open(marker))
    if old!=json.loads(json.dumps(cfg)):
        diff={k:(old.get(k),cfg.get(k)) for k in set(old)|set(cfg) if old.get(k)!=cfg.get(k)}
        raise SystemExit(f"{out_dir} was produced with different settings {diff}.\n"
 f"Use a new --out folder for the new settings (or --force to overwrite everything).")
    return True

def mark_done(out_dir:Path,cfg:dict,args):
    if not args.dry_run:
        out_dir.mkdir(parents=True,exist_ok=True)
        (out_dir/"stage_done.json").write_text(json.dumps(cfg,indent=1,default=str))

def find_best(detector_dir:Path):
    hits=sorted(detector_dir.glob("**/weights/best.pt"))
    return hits[-1] if hits else None

def main():
    args=parse_args()
    root=Path(args.root).resolve()
    out=Path(args.out).resolve()
    cv_dir=root/"cv"
    if not (cv_dir/"folds.json").exists():
        raise SystemExit(f"{cv_dir}/folds.json not found -- run make_group_split.py first.")
    k_total=json.load(open(cv_dir/"folds.json"))["k"]
    folds=list(range(k_total)) if args.folds=="all" else [int(x) for x in args.folds.split(",")]
    stages={s.strip() for s in args.stages.split(",")}
    methods=[m.strip() for m in args.methods.split(",") if m.strip()]
    bad=[m for m in methods if m not in METHODS]
    if bad:
        raise SystemExit(f"Unknown method(s) {bad}; choose from {list(METHODS)}")
    seeds=[int(s) for s in args.seeds.split(",")]
    py=sys.executable
    dev=["--device",args.device] if args.device is not None else []
    out.mkdir(parents=True,exist_ok=True)
    (out/"run_config.json").write_text(json.dumps(vars(args),indent=2))
    for f in folds:
        data=cv_dir/f"fold{f}"/"data.yaml"
        fold_out=out/f"fold{f}"
        det_dir=fold_out/"detector"
        print(f"\n{'=' * 78}\nFOLD {f}   data: {data}\n{'=' * 78}")
        det_cfg={"det_args":args.det_args,"data":str(data)}
        det_done=det_dir.exists() and find_best(det_dir) is not None and \
            stage_ok(det_dir,det_cfg,"stage_done.json",args)
        best=find_best(det_dir) if det_done else None
        if "detector" in stages:
            if not det_done:
                ok=run([py,"train_yolov8.py","--data",data,"--project",det_dir,"--name","train",
                          *shlex.split(args.det_args),*dev],fold_out/"logs"/"detector_train.log",args,
                         f"fold {f}: train detector")
                best=find_best(det_dir)
                if ok and best is not None:
                    mark_done(det_dir,det_cfg,args)
                    for stale in ("detector_metrics","patches","eval","mapdrop"):
                        if (fold_out/stale).exists():
                            shutil.rmtree(fold_out/stale)
            else:
                print(f"fold {f}: detector already trained with these settings -> {best}")
            if best is None and args.dry_run:
                best=Path("<best.pt>")
            if best is not None or args.dry_run:
                for split in ("val","test"):
                    mjson=fold_out/"detector_metrics"/f"{split}_metrics.json"
                    if mjson.exists() and not args.force:
                        continue
                    run([py,"evaluate_yolov8.py","--data",data,"--weights",best,"--split",split,
                         "--no-predictions","--project",det_dir,"--name",f"eval_{split}",
                         "--out-dir",fold_out/"detector_metrics",*dev],
                        fold_out/"logs"/f"detector_eval_{split}.log",args,
                        f"fold {f}: evaluate detector on {'held-out photos' if split == 'val' else 'official test'}")
        if best is None and not args.dry_run:
            print(f"fold {f}: no COMPLETED detector for these settings; skipping patch stages.")
            continue
        best=best or Path("<best.pt>")
        det_stamp=best.stat().st_mtime_ns if best.exists() else None
        for m in methods:
            for s in seeds:
                tag=f"{m}_s{s}"
                patch_dir=fold_out/"patches"/tag
                patch_png=patch_dir/"patch_final.png"
                p_cfg={"detector":str(best),"detector_mtime":det_stamp,"method":m,"corrupt":METHODS[m],"seed":s,
                         "target":args.target_class,"patch_scale":args.patch_scale,
                         "epochs":args.patch_epochs,"ipe":args.instances_per_epoch,"extra":args.patch_args}
                if "patch" in stages and not stage_ok(patch_dir,p_cfg,"patch_final.png",args):
                    cmd=[py,"patch_train.py","--data",data,"--weights",best,"--split","train",
                           "--target-class",args.target_class,"--patch-scale",args.patch_scale,
                           "--epochs",args.patch_epochs,"--instances-per-epoch",args.instances_per_epoch,
                           "--seed",s,"--output-dir",patch_dir,*dev,*shlex.split(args.patch_args)]
                    if METHODS[m]:
                        cmd+=["--corrupt",METHODS[m]]
                    if run(cmd,fold_out/"logs"/f"patch_train_{tag}.log",args,f"fold {f}: train patch {tag}"):
                        mark_done(patch_dir,p_cfg,args)
                if not args.dry_run and not stage_ok(patch_dir,p_cfg,"patch_final.png",args,honor_force=False):
                    continue
                e_cfg={**p_cfg,"max_files_per_photo":args.max_files_per_photo,"eval_seed":args.eval_seed}
                eval_dir=fold_out/"eval"/tag
                if "eval" in stages and not stage_ok(eval_dir,e_cfg,"attack_summary.json",args):
                    ok=run([py,"patch_evaluate.py","--data",data,"--weights",best,"--patch",patch_png,
                         "--target-class",args.target_class,"--split","val","--patch-scale",args.patch_scale,
                         "--corruptions","all","--max-files-per-photo",args.max_files_per_photo,
                         "--seed",args.eval_seed,"--save-examples",4,"--output-dir",eval_dir,*dev],
                        fold_out/"logs"/f"patch_eval_{tag}.log",args,f"fold {f}: evaluate patch {tag}")
                    if ok:
                        mark_done(eval_dir,e_cfg,args)
                md_dir=fold_out/"mapdrop"/tag
                if "mapdrop" in stages and not stage_ok(md_dir,e_cfg,"mapdrop_summary.json",args):
                    ok=run([py,"patch_map_drop.py","--data",data,"--weights",best,"--patch",patch_png,
                         "--target-class",args.target_class,"--split","val","--patch-scale",args.patch_scale,
                         "--max-files-per-photo",args.max_files_per_photo,"--seed",args.eval_seed,
                         "--output-dir",md_dir,*dev],
                        fold_out/"logs"/f"mapdrop_{tag}.log",args,f"fold {f}: mAP drop {tag}")
                    if ok:
                        mark_done(md_dir,e_cfg,args)
    print("\nAll requested jobs finished. Next: python cv_analyze.py --out",out)

if __name__=="__main__":
    main()
