#!/usr/bin/env bash
# 3-D stage on all folds (run from src/; every step is resumable). Main commands:
#   check | preview | smoke | all | status | report            3-D CPA none/light/shadow/both + black-box detectors
#   ablation | ablation_report                                  patch size, seeds, corruption probability
#   diversity | diversity_report                                paper-like fixed-view training
#   transfer | transfer_report                                  4x4 transfer matrix (cpa_none, cpa_ls)
#   npa | npa_report | npa_tau | npa_tau_report                 NPA (BigGAN), pixel baseline, tau / class study
#   natural | natural2 | natural2_report                        sphere latent, coarse latent, saturation penalty,
#                                                               naturalness early-stop, AdvCaT camouflage (1 m, 3 m)
#   npa_sweep | trnat | sweep_trnat_report                      realism-attack curve, natural-NPA transfer
# Env: ROOT (yolo_dataset with cv/), OUT (default cpa3d_runs), SURR_RUN (2-D fold detectors), BIGGAN (G_ema.pth).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.." # src/

ROOT="${ROOT:-../data/Aerial Image Dataset/yolo_dataset}"
SURR_RUN="${SURR_RUN:-cv_runs_e80}" # surrogate detectors: $SURR_RUN/fold{k}/detector
OUT="${OUT:-cpa3d_runs}"
case "$OUT" in /*) ;; *) OUT="$(pwd)/$OUT" ;; esac # Ultralytics puts relative projects under runs/detect
PY="${PY:-python}"
TARGET="${TARGET:-Tank}"
TEX_EPOCHS="${TEX_EPOCHS:-10}" # paper: 10 epochs, SWA from epoch 6
TGT_COMMON="${TGT_COMMON:---epochs 15 --imgsz 960 --patience 5 --workers 4}"
TEX_EXTRA="${TEX_EXTRA:-}" # extra cpa_train.py args (e.g. --max-files-per-photo 12)
EVAL_EXTRA="${EVAL_EXTRA:-}" # extra cpa_eval.py args
declare -A TGT=(
  [yolov8m]="--model yolov8m.pt --batch 16 --freeze 10 $TGT_COMMON"
  [yolo11s]="--model yolo11s.pt --batch 16 --freeze 10 $TGT_COMMON"
  [rtdetr]="--model rtdetr-l.pt --batch 8 $TGT_COMMON"
)
declare -A CORR=([cpa_none]="" [cpa_light]="light" [cpa_shadow]="shadow" [cpa_ls]="light,shadow")
METHODS="cpa_none cpa_light cpa_shadow cpa_ls"

surrogate() { ls "$SURR_RUN/fold$1"/detector/*/weights/best.pt 2>/dev/null | head -1; }

fold_chain() { # fold_chain <fold> -- everything for one fold, in order
  local k=$1 data="$ROOT/cv/fold$1/data.yaml" fo="$OUT/fold$1" s
  s=$(surrogate "$k"); [ -n "$s" ] || { echo "fold $k: no surrogate in $SURR_RUN"; return 1; }
  mkdir -p "$fo/logs"
  for m in $METHODS; do
    if [ -f "$fo/textures/$m/stage_done.json" ]; then echo "fold $k $m: done"; continue; fi
    echo "fold $k: texture $m  $(date +%T)"
    $PY "$HERE/cpa_train.py" --data "$data" --weights "$s" --target-class "$TARGET" \
        --corrupt "${CORR[$m]}" --epochs "$TEX_EPOCHS" --mesh "$HERE/aux_mesh.npz" \
        --out "$fo/textures/$m" $TEX_EXTRA > "$fo/logs/tex_$m.log" 2>&1 || { echo "FAILED texture $m fold $k"; return 1; }
  done
  for t in yolov8m yolo11s rtdetr; do
    if [ -f "$fo/targets/$t/done" ]; then echo "fold $k $t: done"; continue; fi
    echo "fold $k: target $t  $(date +%T)"
    rm -rf "$fo/targets/$t"
    $PY train_yolov8.py --data "$data" ${TGT[$t]} --project "$fo/targets/$t" --name train \
        > "$fo/logs/target_$t.log" 2>&1 || { echo "FAILED target $t fold $k"; return 1; }
    [ -f "$fo/targets/$t/train/weights/best.pt" ] || { echo "FAILED target $t fold $k (no best.pt)"; return 1; }
    $PY evaluate_yolov8.py --data "$data" --weights "$fo/targets/$t/train/weights/best.pt" --split val \
        --no-predictions --project "$fo/targets/$t" --name eval_val --out-dir "$fo/targets/$t/metrics" > "$fo/logs/target_${t}_eval.log" 2>&1 || true
    touch "$fo/targets/$t/done"
  done
  if [ -f "$fo/eval/stage_done.json" ]; then echo "fold $k eval: done"; return 0; fi
  echo "fold $k: eval  $(date +%T)"
  local meth=() dets=("surrogate=$s")
  for m in $METHODS; do meth+=("$m=$fo/textures/$m"); done
  for t in yolov8m yolo11s rtdetr; do dets+=("$t=$fo/targets/$t/train/weights/best.pt"); done
  $PY "$HERE/cpa_eval.py" --data "$data" --split val --target-class "$TARGET" --mesh "$HERE/aux_mesh.npz" \
      --methods "${meth[@]}" --detectors "${dets[@]}" --out "$fo/eval" $EVAL_EXTRA > "$fo/logs/eval.log" 2>&1 \
      || { echo "FAILED eval fold $k"; return 1; }
  echo "fold $k: ALL DONE $(date +%T)"
}

ABL_METHODS="cpa_none_s1 cpa_ls_s1 cpa_ls_p1 cpa_none_1p5 cpa_ls_1p5 cpa_none_2m cpa_ls_2m"
abl_args() {
  case $1 in
    cpa_none_s1)  echo "--corrupt= --seed 1" ;;
    cpa_ls_s1)    echo "--corrupt=light,shadow --seed 1" ;;
    cpa_ls_p1)    echo "--corrupt=light,shadow --corrupt-prob 1.0" ;;
    cpa_none_1p5) echo "--corrupt= --patch-m 1.5" ;;
    cpa_ls_1p5)   echo "--corrupt=light,shadow --patch-m 1.5" ;;
    cpa_none_2m)  echo "--corrupt= --patch-m 2.0" ;;
    cpa_ls_2m)    echo "--corrupt=light,shadow --patch-m 2.0" ;;
  esac
}

abl_eval() { # abl_eval <fold> <eval dir> <patch m> <method>...
  local k=$1 ed=$2 pm=$3; shift 3
  local data="$ROOT/cv/fold$k/data.yaml" fo="$OUT/fold$k" s meth=() dets
  s=$(surrogate "$k"); dets=("surrogate=$s")
  if [ -f "$fo/$ed/stage_done.json" ]; then echo "fold $k $ed: done"; return 0; fi
  for m in "$@"; do meth+=("$m=$fo/textures/$m"); done
  for t in yolov8m yolo11s rtdetr; do dets+=("$t=$fo/targets/$t/train/weights/best.pt"); done
  echo "fold $k: $ed  $(date +%T)"
  $PY "$HERE/cpa_eval.py" --data "$data" --split val --target-class "$TARGET" --mesh "$HERE/aux_mesh.npz" \
      --patch-m "$pm" --methods "${meth[@]}" --detectors "${dets[@]}" --out "$fo/$ed" $EVAL_EXTRA \
      > "$fo/logs/$ed.log" 2>&1 || { echo "FAILED $ed fold $k"; return 1; }
}

abl_chain() { # abl_chain <fold>
  local k=$1 data="$ROOT/cv/fold$1/data.yaml" fo="$OUT/fold$1" s
  s=$(surrogate "$k"); [ -n "$s" ] || { echo "fold $k: no surrogate"; return 1; }
  for t in yolov8m yolo11s rtdetr; do
    [ -f "$fo/targets/$t/done" ] || { echo "fold $k: target $t missing"; return 1; }; done
  for m in cpa_none cpa_ls; do
    [ -f "$fo/textures/$m/stage_done.json" ] || { echo "fold $k: $m missing"; return 1; }; done
  for m in $ABL_METHODS; do
    if [ -f "$fo/textures/$m/stage_done.json" ]; then echo "fold $k $m: done"; continue; fi
    echo "fold $k: texture $m  $(date +%T)"
    $PY "$HERE/cpa_train.py" --data "$data" --weights "$s" --target-class "$TARGET" $(abl_args $m) \
        --epochs "$TEX_EPOCHS" --mesh "$HERE/aux_mesh.npz" --out "$fo/textures/$m" $TEX_EXTRA \
        > "$fo/logs/tex_$m.log" 2>&1 || { echo "FAILED texture $m fold $k"; return 1; }
  done
  abl_eval $k eval_abl_1m  1.0 cpa_none cpa_ls cpa_none_s1 cpa_ls_s1 cpa_ls_p1 || return 1
  abl_eval $k eval_abl_1p5 1.5 cpa_none_1p5 cpa_ls_1p5 || return 1
  abl_eval $k eval_abl_2m  2.0 cpa_none_2m cpa_ls_2m || return 1
  echo "fold $k: ABLATION DONE $(date +%T)"
}

BIGGAN="${BIGGAN:-$(pwd)/pretrained/generators/BigGAN/G_ema.pth}" # from the authors' pretrained_biggan.tar
GAN_CLASS="${GAN_CLASS:-259}"
NPA_EXTRA="${NPA_EXTRA:-}"

npa_chain() { # npa_chain <fold>
  local k=$1 data="$ROOT/cv/fold$1/data.yaml" fo="$OUT/fold$1" s meth dets
  s=$(surrogate "$k"); [ -n "$s" ] || { echo "fold $k: no surrogate"; return 1; }
  for t in yolov8m yolo11s rtdetr; do
    [ -f "$fo/targets/$t/done" ] || { echo "fold $k: target $t missing"; return 1; }; done
  mkdir -p "$fo/logs"
  for m in npa pixel; do
    if [ -f "$fo/textures/$m/stage_done.json" ]; then echo "fold $k $m: done"; continue; fi
    echo "fold $k: $m  $(date +%T)"
    $PY "$HERE/npa_train.py" --mode $m --data "$data" --weights "$s" --target-class "$TARGET" \
        --biggan-weights "$BIGGAN" --gan-class "$GAN_CLASS" --mesh "$HERE/aux_mesh.npz" \
        --out "$fo/textures/$m" $NPA_EXTRA > "$fo/logs/tex_$m.log" 2>&1 || { echo "FAILED $m fold $k"; return 1; }
  done
  if [ -f "$fo/eval_npa/stage_done.json" ]; then echo "fold $k eval_npa: done"; return 0; fi
  meth=("cpa_none=$fo/textures/cpa_none" "cpa_ls=$fo/textures/cpa_ls" "npa=$fo/textures/npa"
        "npa_init=$fo/textures/npa/patch_init.npy" "pixel=$fo/textures/pixel")
  dets=("surrogate=$s"); for t in yolov8m yolo11s rtdetr; do dets+=("$t=$fo/targets/$t/train/weights/best.pt"); done
  echo "fold $k: eval_npa  $(date +%T)"
  $PY "$HERE/cpa_eval.py" --data "$data" --split val --target-class "$TARGET" --mesh "$HERE/aux_mesh.npz" \
      --methods "${meth[@]}" --detectors "${dets[@]}" --out "$fo/eval_npa" $EVAL_EXTRA \
      > "$fo/logs/eval_npa.log" 2>&1 || { echo "FAILED eval_npa fold $k"; return 1; }
  echo "fold $k: NPA DONE $(date +%T)"
}

NPA_CLASSES="${NPA_CLASSES:-977 958}" # 977 sandbar (terrain), 958 hay (vegetation)
GPUS="${GPUS:-0 1 2 3}" # GPUs to spread the folds over

npacls_chain() { # npacls_chain <fold>
  local k=$1 data="$ROOT/cv/fold$1/data.yaml" fo="$OUT/fold$1" s c meth dets w
  s=$(surrogate "$k"); [ -n "$s" ] || { echo "fold $k: no surrogate"; return 1; }
  mkdir -p "$fo/logs"
  for c in $NPA_CLASSES; do
    if [ -f "$fo/textures/npa_c$c/stage_done.json" ]; then echo "fold $k npa_c$c: done"; continue; fi
    echo "fold $k: npa_c$c  $(date +%T)"
    $PY "$HERE/npa_train.py" --mode npa --data "$data" --weights "$s" --target-class "$TARGET" \
        --biggan-weights "$BIGGAN" --gan-class "$c" --mesh "$HERE/aux_mesh.npz" \
        --out "$fo/textures/npa_c$c" $NPA_EXTRA > "$fo/logs/tex_npa_c$c.log" 2>&1 || { echo "FAILED npa_c$c fold $k"; return 1; }
  done
  for w in $(seq 1 120); do # the class-259 NPA + pixel run of this fold must be finished
    [ -f "$fo/textures/npa/stage_done.json" ] && [ -f "$fo/textures/pixel/stage_done.json" ] && break
    [ $w = 1 ] && echo "fold $k: waiting for npa/pixel"; sleep 60
  done
  [ -f "$fo/textures/pixel/stage_done.json" ] || { echo "FAILED fold $k: npa/pixel never finished"; return 1; }
  if [ -f "$fo/eval_npa_cls/stage_done.json" ]; then echo "fold $k eval_npa_cls: done"; return 0; fi
  meth=("npa=$fo/textures/npa" "npa_init=$fo/textures/npa/patch_init.npy")
  for c in $NPA_CLASSES; do meth+=("npa_c$c=$fo/textures/npa_c$c" "npa_c${c}_init=$fo/textures/npa_c$c/patch_init.npy"); done
  meth+=("pixel=$fo/textures/pixel" "cpa_none=$fo/textures/cpa_none")
  dets=("surrogate=$s"); for t in yolov8m yolo11s rtdetr; do dets+=("$t=$fo/targets/$t/train/weights/best.pt"); done
  echo "fold $k: eval_npa_cls  $(date +%T)"
  $PY "$HERE/cpa_eval.py" --data "$data" --split val --target-class "$TARGET" --mesh "$HERE/aux_mesh.npz" \
      --methods "${meth[@]}" --detectors "${dets[@]}" --out "$fo/eval_npa_cls" $EVAL_EXTRA \
      > "$fo/logs/eval_npa_cls.log" 2>&1 || { echo "FAILED eval_npa_cls fold $k"; return 1; }
  echo "fold $k: NPA CLASSES DONE $(date +%T)"
}

launch_spread() { # launch_spread <chain function> <log prefix>: folds 0-4 round-robin over $GPUS
  local fn=$1 name=$2 g i=0; local -a gl=($GPUS); declare -A F=()
  for k in 0 1 2 3 4; do g=${gl[$((i % ${#gl[@]}))]}; F[$g]="${F[$g]:-} $k"; i=$((i + 1)); done
  mkdir -p "$OUT/logs"
  for g in "${!F[@]}"; do
    CUDA_VISIBLE_DEVICES=$g nohup bash -c "source '$HERE/run_cpa3d.sh' lib; for k in ${F[$g]}; do $fn \$k || true; done" \
        > "$OUT/logs/${name}_gpu$g.log" 2>&1 &
    echo "$name worker GPU $g: folds${F[$g]}  (log $OUT/logs/${name}_gpu$g.log)"
  done
}

TR_SURR="${TR_SURR:-yolov8m yolo11s rtdetr}" # extra surrogates (yolov8s = the main runs)
TR_ATTACKS="${TR_ATTACKS:-cpa_none cpa_ls npa}"
TR_CONDS="${TR_CONDS:-none}" # test conditions for the matrix (paper: clean images)
TR_NPA_MAIN="${TR_NPA_MAIN:-npa}" # YOLOv8s NPA patch used in the matrix (e.g. npa_c977_t4)

tr_chain() { # tr_chain <fold>
  local k=$1 data="$ROOT/cv/fold$1/data.yaml" fo="$OUT/fold$1" s sw a d meth dets w
  s=$(surrogate "$k"); [ -n "$s" ] || { echo "fold $k: no surrogate"; return 1; }
  mkdir -p "$fo/logs"
  for sn in $TR_SURR; do
    sw="$fo/targets/$sn/train/weights/best.pt"
    [ -f "$sw" ] || { echo "FAILED fold $k: no $sn detector"; return 1; }
    for a in $TR_ATTACKS; do
      d="$fo/textures/${a}__$sn"
      if [ -f "$d/stage_done.json" ]; then echo "fold $k ${a}__$sn: done"; continue; fi
      echo "fold $k: ${a}__$sn  $(date +%T)"
      case $a in
        cpa_none) $PY "$HERE/cpa_train.py" --data "$data" --weights "$sw" --target-class "$TARGET" --corrupt= \
                      --epochs "$TEX_EPOCHS" --mesh "$HERE/aux_mesh.npz" --out "$d" $TEX_EXTRA ;;
        cpa_ls)   $PY "$HERE/cpa_train.py" --data "$data" --weights "$sw" --target-class "$TARGET" --corrupt=light,shadow \
                      --epochs "$TEX_EPOCHS" --mesh "$HERE/aux_mesh.npz" --out "$d" $TEX_EXTRA ;;
        npa)      $PY "$HERE/npa_train.py" --mode npa --data "$data" --weights "$sw" --target-class "$TARGET" \
                      --biggan-weights "$BIGGAN" --gan-class "$GAN_CLASS" --mesh "$HERE/aux_mesh.npz" --out "$d" $NPA_EXTRA ;;
      esac > "$fo/logs/tex_${a}__$sn.log" 2>&1 || { echo "FAILED ${a}__$sn fold $k"; return 1; }
    done
  done
  for w in $(seq 1 180); do # main-run npa of this fold must exist
    case " $TR_ATTACKS " in *" npa "*) ;; *) break ;; esac
    [ -f "$fo/textures/$TR_NPA_MAIN/stage_done.json" ] && break
    [ $w = 1 ] && echo "fold $k: waiting for textures/$TR_NPA_MAIN"; sleep 60
  done
  if [ -f "$fo/eval_transfer/stage_done.json" ]; then echo "fold $k eval_transfer: done"; return 0; fi
  meth=()
  for a in $TR_ATTACKS; do
    if [ "$a" = npa ]; then meth+=("$a@yolov8s=$fo/textures/$TR_NPA_MAIN"); else meth+=("$a@yolov8s=$fo/textures/$a"); fi
    for sn in $TR_SURR; do meth+=("$a@$sn=$fo/textures/${a}__$sn"); done
  done
  dets=("surrogate=$s"); for t in yolov8m yolo11s rtdetr; do dets+=("$t=$fo/targets/$t/train/weights/best.pt"); done
  echo "fold $k: eval_transfer  $(date +%T)"
  $PY "$HERE/cpa_eval.py" --data "$data" --split val --target-class "$TARGET" --mesh "$HERE/aux_mesh.npz" \
      --conditions "$(echo $TR_CONDS | tr ' ' ',')" --methods "${meth[@]}" --detectors "${dets[@]}" \
      --out "$fo/eval_transfer" $EVAL_EXTRA > "$fo/logs/eval_transfer.log" 2>&1 || { echo "FAILED eval_transfer fold $k"; return 1; }
  echo "fold $k: TRANSFER DONE $(date +%T)"
}

NPA_GRID="${NPA_GRID:-259:2 259:4 977:4 958:4}" # class:tau  (main NPA run = 259:10)

npatau_chain() { # npatau_chain <fold>
  local k=$1 data="$ROOT/cv/fold$1/data.yaml" fo="$OUT/fold$1" s c t ct d meth dets w
  s=$(surrogate "$k"); [ -n "$s" ] || { echo "fold $k: no surrogate"; return 1; }
  mkdir -p "$fo/logs"
  for ct in $NPA_GRID; do
    c=${ct%%:*}; t=${ct##*:}; d="$fo/textures/npa_c${c}_t$t"
    if [ -f "$d/stage_done.json" ]; then echo "fold $k npa_c${c}_t$t: done"; continue; fi
    echo "fold $k: npa_c${c}_t$t  $(date +%T)"
    $PY "$HERE/npa_train.py" --mode npa --data "$data" --weights "$s" --target-class "$TARGET" \
        --biggan-weights "$BIGGAN" --gan-class "$c" --tau "$t" --mesh "$HERE/aux_mesh.npz" \
        --out "$d" $NPA_EXTRA > "$fo/logs/tex_npa_c${c}_t$t.log" 2>&1 || { echo "FAILED npa_c${c}_t$t fold $k"; return 1; }
  done
  for w in $(seq 1 120); do
    [ -f "$fo/textures/npa/stage_done.json" ] && [ -f "$fo/textures/pixel/stage_done.json" ] && break
    [ $w = 1 ] && echo "fold $k: waiting for npa/pixel"; sleep 60
  done
  if [ -f "$fo/eval_npa_tau/stage_done.json" ]; then echo "fold $k eval_npa_tau: done"; return 0; fi
  meth=("npa=$fo/textures/npa" "npa_init=$fo/textures/npa/patch_init.npy")
  for ct in $NPA_GRID; do c=${ct%%:*}; t=${ct##*:}; meth+=("npa_c${c}_t$t=$fo/textures/npa_c${c}_t$t"); done
  for c in $(for ct in $NPA_GRID; do echo ${ct%%:*}; done | sort -u); do
    [ "$c" = 259 ] && continue
    d=$(ls -d $fo/textures/npa_c${c}_t* | head -1); meth+=("npa_c${c}_init=$d/patch_init.npy"); done
  meth+=("pixel=$fo/textures/pixel" "cpa_none=$fo/textures/cpa_none")
  dets=("surrogate=$s"); for t in yolov8m yolo11s rtdetr; do dets+=("$t=$fo/targets/$t/train/weights/best.pt"); done
  echo "fold $k: eval_npa_tau  $(date +%T)"
  $PY "$HERE/cpa_eval.py" --data "$data" --split val --target-class "$TARGET" --mesh "$HERE/aux_mesh.npz" \
      --methods "${meth[@]}" --detectors "${dets[@]}" --out "$fo/eval_npa_tau" $EVAL_EXTRA \
      > "$fo/logs/eval_npa_tau.log" 2>&1 || { echo "FAILED eval_npa_tau fold $k"; return 1; }
  echo "fold $k: NPA TAU DONE $(date +%T)"
}

NAT_VARIANTS="${NAT_VARIANTS:-camo_wood camo_auto npa_sph}"
nat_args() {
  case $1 in
    camo_wood) echo "--mode camo --palette woodland" ;;
    camo_auto) echo "--mode camo --palette auto" ;;
    npa_sph)   echo "--mode npa --gan-class 977 --tau 3 --z-proj sphere" ;;
  esac
}

nat_chain() { # nat_chain <fold>
  local k=$1 data="$ROOT/cv/fold$1/data.yaml" fo="$OUT/fold$1" s v d meth dets
  s=$(surrogate "$k"); [ -n "$s" ] || { echo "fold $k: no surrogate"; return 1; }
  mkdir -p "$fo/logs"
  for v in $NAT_VARIANTS; do
    d="$fo/textures/$v"
    if [ -f "$d/stage_done.json" ]; then echo "fold $k $v: done"; continue; fi
    rm -f "$d/viewbank.npz"
    echo "fold $k: $v  $(date +%T)"
    $PY "$HERE/npa_train.py" $(nat_args $v) --data "$data" --weights "$s" --target-class "$TARGET" \
        --biggan-weights "$BIGGAN" --mesh "$HERE/aux_mesh.npz" --out "$d" $NPA_EXTRA \
        > "$fo/logs/tex_$v.log" 2>&1 || { echo "FAILED $v fold $k"; return 1; }
  done
  if [ -f "$fo/eval_natural/stage_done.json" ]; then echo "fold $k eval_natural: done"; return 0; fi
  meth=()
  for v in $NAT_VARIANTS; do
    meth+=("$v=$fo/textures/$v"); case $v in camo*) meth+=("${v}_init=$fo/textures/$v/patch_init.npy") ;; esac
  done
  for v in npa_c977_t4 npa pixel cpa_none; do [ -f "$fo/textures/$v/stage_done.json" ] && meth+=("$v=$fo/textures/$v"); done
  dets=("surrogate=$s"); for t in yolov8m yolo11s rtdetr; do dets+=("$t=$fo/targets/$t/train/weights/best.pt"); done
  echo "fold $k: eval_natural  $(date +%T)"
  $PY "$HERE/cpa_eval.py" --data "$data" --split val --target-class "$TARGET" --mesh "$HERE/aux_mesh.npz" \
      --methods "${meth[@]}" --detectors "${dets[@]}" --out "$fo/eval_natural" $EVAL_EXTRA \
      > "$fo/logs/eval_natural.log" 2>&1 || { echo "FAILED eval_natural fold $k"; return 1; }
  echo "fold $k: NATURAL DONE $(date +%T)"
}

DIV_VARIANTS="cpa_none_fix cpa_ls_fix cpa_none_fix_s1 cpa_ls_fix_s1"
div_args() {
  case $1 in
    cpa_none_fix)    echo "--corrupt= --fixed-views" ;;
    cpa_ls_fix)      echo "--corrupt=light,shadow --fixed-views" ;;
    cpa_none_fix_s1) echo "--corrupt= --fixed-views --seed 1" ;;
    cpa_ls_fix_s1)   echo "--corrupt=light,shadow --fixed-views --seed 1" ;;
  esac
}

div_chain() { # div_chain <fold>
  local k=$1 data="$ROOT/cv/fold$1/data.yaml" fo="$OUT/fold$1" s v d meth dets
  s=$(surrogate "$k"); [ -n "$s" ] || { echo "fold $k: no surrogate"; return 1; }
  for v in cpa_none cpa_ls cpa_none_s1 cpa_ls_s1; do
    [ -f "$fo/textures/$v/stage_done.json" ] || { echo "FAILED fold $k: $v missing (main/ablation run)"; return 1; }; done
  mkdir -p "$fo/logs"
  for v in $DIV_VARIANTS; do
    d="$fo/textures/$v"
    if [ -f "$d/stage_done.json" ]; then echo "fold $k $v: done"; continue; fi
    rm -f "$d/viewbank.npz"
    echo "fold $k: $v  $(date +%T)"
    $PY "$HERE/cpa_train.py" --data "$data" --weights "$s" --target-class "$TARGET" $(div_args $v) \
        --epochs "$TEX_EPOCHS" --mesh "$HERE/aux_mesh.npz" --out "$d" $TEX_EXTRA \
        > "$fo/logs/tex_$v.log" 2>&1 || { echo "FAILED $v fold $k"; return 1; }
  done
  if [ -f "$fo/eval_diversity/stage_done.json" ]; then echo "fold $k eval_diversity: done"; return 0; fi
  meth=(); for v in $DIV_VARIANTS cpa_none cpa_ls cpa_none_s1 cpa_ls_s1; do meth+=("$v=$fo/textures/$v"); done
  dets=("surrogate=$s"); for t in yolov8m yolo11s rtdetr; do dets+=("$t=$fo/targets/$t/train/weights/best.pt"); done
  echo "fold $k: eval_diversity  $(date +%T)"
  $PY "$HERE/cpa_eval.py" --data "$data" --split val --target-class "$TARGET" --mesh "$HERE/aux_mesh.npz" \
      --methods "${meth[@]}" --detectors "${dets[@]}" --out "$fo/eval_diversity" $EVAL_EXTRA \
      > "$fo/logs/eval_diversity.log" 2>&1 || { echo "FAILED eval_diversity fold $k"; return 1; }
  echo "fold $k: DIVERSITY DONE $(date +%T)"
}

NAT2_1M="npa_coarse npa_satpen npa_stop"
NAT2_3M="camo_wood_3m camo_auto_3m"
nat2_args() {
  local base="--mode npa --gan-class 977 --tau 3 --z-proj sphere"
  case $1 in
    npa_coarse)   echo "$base --z-chunks 0,1,2" ;;
    npa_satpen)   echo "$base --sat-weight 1.0" ;;
    npa_stop)     echo "$base --natural-stop 20" ;;
    camo_wood_3m) echo "--mode camo --palette woodland --patch-m 3.0" ;;
    camo_auto_3m) echo "--mode camo --palette auto --patch-m 3.0" ;;
  esac
}

nat2_chain() { # nat2_chain <fold>
  local k=$1 data="$ROOT/cv/fold$1/data.yaml" fo="$OUT/fold$1" s v d meth dets
  s=$(surrogate "$k"); [ -n "$s" ] || { echo "fold $k: no surrogate"; return 1; }
  mkdir -p "$fo/logs"
  for v in $NAT2_1M $NAT2_3M; do
    d="$fo/textures/$v"
    if [ -f "$d/stage_done.json" ]; then echo "fold $k $v: done"; continue; fi
    rm -f "$d/viewbank.npz"
    echo "fold $k: $v  $(date +%T)"
    $PY "$HERE/npa_train.py" $(nat2_args $v) --data "$data" --weights "$s" --target-class "$TARGET" \
        --biggan-weights "$BIGGAN" --mesh "$HERE/aux_mesh.npz" --out "$d" $NPA_EXTRA \
        > "$fo/logs/tex_$v.log" 2>&1 || { echo "FAILED $v fold $k"; return 1; }
  done
  dets=("surrogate=$s"); for t in yolov8m yolo11s rtdetr; do dets+=("$t=$fo/targets/$t/train/weights/best.pt"); done
  if [ ! -f "$fo/eval_natural2/stage_done.json" ]; then
    meth=(); for v in $NAT2_1M npa_sph npa_c977_t4 cpa_none; do meth+=("$v=$fo/textures/$v"); done
    echo "fold $k: eval_natural2  $(date +%T)"
    $PY "$HERE/cpa_eval.py" --data "$data" --split val --target-class "$TARGET" --mesh "$HERE/aux_mesh.npz" \
        --methods "${meth[@]}" --detectors "${dets[@]}" --out "$fo/eval_natural2" $EVAL_EXTRA \
        > "$fo/logs/eval_natural2.log" 2>&1 || { echo "FAILED eval_natural2 fold $k"; return 1; }
  fi
  if [ ! -f "$fo/eval_camo3m/stage_done.json" ]; then
    meth=(); for v in $NAT2_3M; do meth+=("$v=$fo/textures/$v" "${v}_init=$fo/textures/$v/patch_init.npy"); done
    echo "fold $k: eval_camo3m  $(date +%T)"
    $PY "$HERE/cpa_eval.py" --data "$data" --split val --target-class "$TARGET" --mesh "$HERE/aux_mesh.npz" \
        --patch-m 3.0 --methods "${meth[@]}" --detectors "${dets[@]}" --out "$fo/eval_camo3m" $EVAL_EXTRA \
        > "$fo/logs/eval_camo3m.log" 2>&1 || { echo "FAILED eval_camo3m fold $k"; return 1; }
  fi
  echo "fold $k: NATURAL2 DONE $(date +%T)"
}

SWEEP_BUDGETS="${SWEEP_BUDGETS:-5 10 20 40 100}"
NPA_NAT_ARGS="--mode npa --gan-class 977 --tau 3 --z-proj sphere --natural-stop 20"

sweep_chain() { # sweep_chain <fold>
  local k=$1 data="$ROOT/cv/fold$1/data.yaml" fo="$OUT/fold$1" s b meth dets
  s=$(surrogate "$k"); [ -n "$s" ] || { echo "fold $k: no surrogate"; return 1; }
  [ -f "$fo/textures/npa_stop/stage_done.json" ] || { echo "FAILED fold $k: npa_stop missing (natural2)"; return 1; }
  $PY "$HERE/npa_sweep.py" --runs "$OUT" --fold "$k" --budgets $SWEEP_BUDGETS > "$fo/logs/npa_sweep.log" 2>&1 \
      || { echo "FAILED npa_sweep fold $k"; return 1; }
  if [ -f "$fo/eval_npa_sweep/stage_done.json" ]; then echo "fold $k eval_npa_sweep: done"; return 0; fi
  meth=("npa_stop_init=$fo/textures/npa_stop/patch_init.npy")
  for b in $SWEEP_BUDGETS; do meth+=("npa_stop$b=$fo/textures/npa_stop$b"); done
  meth+=("cpa_none=$fo/textures/cpa_none")
  dets=("surrogate=$s"); for t in yolov8m yolo11s rtdetr; do dets+=("$t=$fo/targets/$t/train/weights/best.pt"); done
  echo "fold $k: eval_npa_sweep  $(date +%T)"
  $PY "$HERE/cpa_eval.py" --data "$data" --split val --target-class "$TARGET" --mesh "$HERE/aux_mesh.npz" \
      --methods "${meth[@]}" --detectors "${dets[@]}" --out "$fo/eval_npa_sweep" $EVAL_EXTRA \
      > "$fo/logs/eval_npa_sweep.log" 2>&1 || { echo "FAILED eval_npa_sweep fold $k"; return 1; }
  echo "fold $k: SWEEP DONE $(date +%T)"
}

trnat_chain() { # trnat_chain <fold>
  local k=$1 data="$ROOT/cv/fold$1/data.yaml" fo="$OUT/fold$1" s sn sw d a meth dets
  s=$(surrogate "$k"); [ -n "$s" ] || { echo "fold $k: no surrogate"; return 1; }
  mkdir -p "$fo/logs"
  for sn in yolov8m yolo11s rtdetr; do
    sw="$fo/targets/$sn/train/weights/best.pt"; d="$fo/textures/npa_stop__$sn"
    if [ -f "$d/stage_done.json" ]; then echo "fold $k npa_stop__$sn: done"; continue; fi
    rm -f "$d/viewbank.npz"
    echo "fold $k: npa_stop__$sn  $(date +%T)"
    $PY "$HERE/npa_train.py" $NPA_NAT_ARGS --data "$data" --weights "$sw" --target-class "$TARGET" \
        --biggan-weights "$BIGGAN" --mesh "$HERE/aux_mesh.npz" --out "$d" $NPA_EXTRA \
        > "$fo/logs/tex_npa_stop__$sn.log" 2>&1 || { echo "FAILED npa_stop__$sn fold $k"; return 1; }
  done
  if [ -f "$fo/eval_transfer_nat/stage_done.json" ]; then echo "fold $k eval_transfer_nat: done"; return 0; fi
  meth=("npa_stop@yolov8s=$fo/textures/npa_stop")
  for sn in yolov8m yolo11s rtdetr; do meth+=("npa_stop@$sn=$fo/textures/npa_stop__$sn"); done
  for a in cpa_none cpa_ls; do
    meth+=("$a@yolov8s=$fo/textures/$a")
    for sn in yolov8m yolo11s rtdetr; do meth+=("$a@$sn=$fo/textures/${a}__$sn"); done
  done
  dets=("surrogate=$s"); for t in yolov8m yolo11s rtdetr; do dets+=("$t=$fo/targets/$t/train/weights/best.pt"); done
  echo "fold $k: eval_transfer_nat  $(date +%T)"
  $PY "$HERE/cpa_eval.py" --data "$data" --split val --target-class "$TARGET" --mesh "$HERE/aux_mesh.npz" \
      --conditions none --methods "${meth[@]}" --detectors "${dets[@]}" --out "$fo/eval_transfer_nat" $EVAL_EXTRA \
      > "$fo/logs/eval_transfer_nat.log" 2>&1 || { echo "FAILED eval_transfer_nat fold $k"; return 1; }
  echo "fold $k: TRANSFER NAT DONE $(date +%T)"
}

case "${1:-}" in
  check)
    ok=1
    for f in cpa_train.py cpa_eval.py cpa_analyze.py render.py place.py corrupt_paper.py generator.py common.py aux_mesh.npz; do
      [ -f "$HERE/$f" ] || { echo "missing cpa3d/$f"; ok=0; }; done
    for f in yolov8m.pt yolo11s.pt rtdetr-l.pt train_yolov8.py evaluate_yolov8.py photo_stats.py dataset_utils.py; do
      [ -f "$f" ] || { echo "missing $f in $(pwd)"; ok=0; }; done
    for k in 0 1 2 3 4; do
      [ -f "$ROOT/cv/fold$k/data.yaml" ] || { echo "missing $ROOT/cv/fold$k/data.yaml"; ok=0; }
      [ -n "$(surrogate $k)" ] || { echo "no surrogate detector for fold $k in $SURR_RUN"; ok=0; }
    done
    $PY -c "import torch, ultralytics, cv2; from ultralytics import RTDETR; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'ultralytics', ultralytics.__version__)" || ok=0
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || echo "(no nvidia-smi)"
    [ $ok = 1 ] && echo "CHECK OK" || { echo "CHECK FAILED"; exit 1; }
    ;;
  preview)
    $PY "$HERE/cpa_preview.py" --data "$ROOT/cv/fold0/data.yaml" --split train --mesh "$HERE/aux_mesh.npz" \
        --n 18 --out cpa3d_preview.jpg
    $PY "$HERE/cpa_preview.py" --data "$ROOT/cv/fold0/data.yaml" --split train --mesh "$HERE/aux_mesh.npz" \
        --n 4 --corruptions --out cpa3d_preview_corruptions.jpg
    ;;
  smoke)
    s=$(surrogate 0); d="$ROOT/cv/fold0/data.yaml"; o=cpa3d_smoke; rm -rf $o
    CUDA_VISIBLE_DEVICES=${GPU:-0} $PY "$HERE/cpa_train.py" --data "$d" --weights "$s" --corrupt light,shadow \
        --epochs 2 --swa-start 1 --max-steps 3 --n-views 16 --mesh "$HERE/aux_mesh.npz" --out $o/tex
    CUDA_VISIBLE_DEVICES=${GPU:-0} $PY "$HERE/cpa_eval.py" --data "$d" --methods smoke=$o/tex --detectors surrogate="$s" \
        --max-files-per-photo 1 --n-views 16 --mesh "$HERE/aux_mesh.npz" --out $o/eval
    echo "SMOKE OK"
    ;;
  all)
    mkdir -p "$OUT/logs"
    for g in 0 1 2 3; do
      folds="$g"; [ $g = 0 ] && folds="0 4"
      CUDA_VISIBLE_DEVICES=$g nohup bash -c "source '$HERE/run_cpa3d.sh' lib; for k in $folds; do fold_chain \$k || true; done" \
          > "$OUT/logs/worker_gpu$g.log" 2>&1 &
      echo "worker GPU $g: folds $folds  (log $OUT/logs/worker_gpu$g.log)"
    done
    ;;
  ablation)
    mkdir -p "$OUT/logs"
    for g in 0 1 2 3; do
      folds="$g"; [ $g = 0 ] && folds="0 4"
      CUDA_VISIBLE_DEVICES=$g nohup bash -c "source '$HERE/run_cpa3d.sh' lib; for k in $folds; do abl_chain \$k || true; done" \
          > "$OUT/logs/ablation_gpu$g.log" 2>&1 &
      echo "ablation worker GPU $g: folds $folds  (log $OUT/logs/ablation_gpu$g.log)"
    done
    ;;
  ablation_status)
    echo "workers running: $( (pgrep -u "${USER:-$(id -un)}" -f 'run_cpa3d.sh. lib' || true) | wc -l)"
    for k in 0 1 2 3 4; do
      t=0; for m in $ABL_METHODS; do [ -f $OUT/fold$k/textures/$m/stage_done.json ] && t=$((t+1)); done
      e=$( (ls $OUT/fold$k/eval_abl_*/stage_done.json 2>/dev/null || true) | wc -l)
      echo "fold $k: ablation textures $t/7  evals $e/3"
    done
    tail -n 2 $OUT/logs/ablation_gpu*.log 2>/dev/null || true
    grep -h FAILED $OUT/logs/ablation_gpu*.log 2>/dev/null || true
    ;;
  ablation_report)
    $PY "$HERE/cpa_ablation.py" --runs "$OUT"
    ;;
  npa_smoke)
    [ -f "$BIGGAN" ] || { echo "BigGAN weights not found: $BIGGAN"; exit 1; }
    s=$(surrogate 0); d="$ROOT/cv/fold0/data.yaml"; o=npa_smoke; rm -rf $o
    for m in npa pixel; do
      CUDA_VISIBLE_DEVICES=${GPU:-0} $PY "$HERE/npa_train.py" --mode $m --data "$d" --weights "$s" \
          --biggan-weights "$BIGGAN" --gan-class "$GAN_CLASS" --epochs 2 --max-steps 3 --n-views 16 \
          --mesh "$HERE/aux_mesh.npz" --out $o/$m || exit 1
    done
    echo "NPA SMOKE OK"
    ;;
  npa)
    [ -f "$BIGGAN" ] || { echo "BigGAN weights not found: $BIGGAN"; exit 1; }
    mkdir -p "$OUT/logs"
    for g in 0 1 2 3; do
      folds="$g"; [ $g = 0 ] && folds="0 4"
      CUDA_VISIBLE_DEVICES=$g nohup bash -c "source '$HERE/run_cpa3d.sh' lib; for k in $folds; do npa_chain \$k || true; done" \
          > "$OUT/logs/npa_gpu$g.log" 2>&1 &
      echo "npa worker GPU $g: folds $folds  (log $OUT/logs/npa_gpu$g.log)"
    done
    ;;
  npa_status)
    echo "workers running: $( (pgrep -u "${USER:-$(id -un)}" -f 'run_cpa3d.sh. lib' || true) | wc -l)"
    for k in 0 1 2 3 4; do
      t=0; for m in npa pixel; do [ -f $OUT/fold$k/textures/$m/stage_done.json ] && t=$((t+1)); done
      e=$([ -f $OUT/fold$k/eval_npa/stage_done.json ] && echo yes || echo no)
      echo "fold $k: npa/pixel $t/2  eval $e"
    done
    tail -n 2 $OUT/logs/npa_gpu*.log 2>/dev/null || true
    grep -h FAILED $OUT/logs/npa_gpu*.log 2>/dev/null || true
    ;;
  npa_report)
    $PY "$HERE/cpa_analyze.py" --runs "$OUT" --eval-dir eval_npa --baseline npa --report-name npa_report
    ;;
  npa_classes)
    [ -f "$BIGGAN" ] || { echo "BigGAN weights not found: $BIGGAN"; exit 1; }
    echo "classes: $NPA_CLASSES   GPUs: $GPUS"
    launch_spread npacls_chain npacls
    ;;
  npa_classes_status)
    echo "workers running: $( (pgrep -u "${USER:-$(id -un)}" -f 'run_cpa3d.sh. lib' || true) | wc -l)"
    for k in 0 1 2 3 4; do
      t=0; n=0; for c in $NPA_CLASSES; do n=$((n+1)); [ -f $OUT/fold$k/textures/npa_c$c/stage_done.json ] && t=$((t+1)); done
      e=$([ -f $OUT/fold$k/eval_npa_cls/stage_done.json ] && echo yes || echo no)
      echo "fold $k: class patches $t/$n  eval $e"
    done
    tail -n 2 $OUT/logs/npacls_gpu*.log 2>/dev/null || true
    grep -h FAILED $OUT/logs/npacls_gpu*.log 2>/dev/null || true
    ;;
  npa_classes_report)
    $PY "$HERE/cpa_analyze.py" --runs "$OUT" --eval-dir eval_npa_cls --baseline npa --report-name npa_classes_report
    ;;
  npa_tau)
    [ -f "$BIGGAN" ] || { echo "BigGAN weights not found: $BIGGAN"; exit 1; }
    echo "grid (class:tau): $NPA_GRID   GPUs: $GPUS"
    launch_spread npatau_chain npatau
    ;;
  npa_tau_status)
    echo "workers running: $( (pgrep -u "${USER:-$(id -un)}" -f 'run_cpa3d.sh. lib' || true) | wc -l)"
    for k in 0 1 2 3 4; do
      t=0; n=0; for ct in $NPA_GRID; do n=$((n+1)); [ -f $OUT/fold$k/textures/npa_c${ct%%:*}_t${ct##*:}/stage_done.json ] && t=$((t+1)); done
      e=$([ -f $OUT/fold$k/eval_npa_tau/stage_done.json ] && echo yes || echo no)
      echo "fold $k: tau patches $t/$n  eval $e"
    done
    tail -n 2 $OUT/logs/npatau_gpu*.log 2>/dev/null || true
    grep -h FAILED $OUT/logs/npatau_gpu*.log 2>/dev/null || true
    ;;
  npa_tau_report)
    $PY "$HERE/npa_naturalness.py" --runs "$OUT"
    $PY "$HERE/cpa_analyze.py" --runs "$OUT" --eval-dir eval_npa_tau --baseline npa --report-name npa_tau_report
    ;;
  natural)
    echo "variants: $NAT_VARIANTS   GPUs: $GPUS"
    launch_spread nat_chain natural
    ;;
  natural_status)
    echo "workers running: $( (pgrep -u "${USER:-$(id -un)}" -f 'run_cpa3d.sh. lib' || true) | wc -l)"
    for k in 0 1 2 3 4; do
      t=0; n=0; for v in $NAT_VARIANTS; do n=$((n+1)); [ -f $OUT/fold$k/textures/$v/stage_done.json ] && t=$((t+1)); done
      e=$([ -f $OUT/fold$k/eval_natural/stage_done.json ] && echo yes || echo no)
      echo "fold $k: natural patches $t/$n  eval $e"
    done
    tail -n 2 $OUT/logs/natural_gpu*.log 2>/dev/null || true
    grep -h FAILED $OUT/logs/natural_gpu*.log 2>/dev/null || true
    ;;
  natural_report)
    $PY "$HERE/npa_naturalness.py" --runs "$OUT"
    $PY "$HERE/cpa_analyze.py" --runs "$OUT" --eval-dir eval_natural --baseline cpa_none --report-name natural_report
    ;;
  natural2)
    [ -f "$BIGGAN" ] || { echo "BigGAN weights not found: $BIGGAN"; exit 1; }
    echo "1 m: $NAT2_1M | 3 m: $NAT2_3M | GPUs: $GPUS"
    launch_spread nat2_chain natural2
    ;;
  natural2_status)
    echo "workers running: $( (pgrep -u "${USER:-$(id -un)}" -f 'run_cpa3d.sh. lib' || true) | wc -l)"
    for k in 0 1 2 3 4; do
      t=0; for v in $NAT2_1M $NAT2_3M; do [ -f $OUT/fold$k/textures/$v/stage_done.json ] && t=$((t+1)); done
      e=0; for e2 in eval_natural2 eval_camo3m; do [ -f $OUT/fold$k/$e2/stage_done.json ] && e=$((e+1)); done
      echo "fold $k: patches $t/5  evals $e/2"
    done
    tail -n 2 $OUT/logs/natural2_gpu*.log 2>/dev/null || true
    grep -h FAILED $OUT/logs/natural2_gpu*.log 2>/dev/null || true
    ;;
  natural2_report)
    $PY "$HERE/npa_naturalness.py" --runs "$OUT"
    $PY "$HERE/cpa_analyze.py" --runs "$OUT" --eval-dir eval_natural2 --baseline npa_sph --report-name natural2_report
    $PY "$HERE/cpa_analyze.py" --runs "$OUT" --eval-dir eval_camo3m --baseline camo_wood_3m --report-name camo3m_report
    ;;
  npa_sweep)
    launch_spread sweep_chain sweep
    ;;
  trnat)
    [ -f "$BIGGAN" ] || { echo "BigGAN weights not found: $BIGGAN"; exit 1; }
    launch_spread trnat_chain trnat
    ;;
  sweep_trnat_status)
    echo "workers running: $( (pgrep -u "${USER:-$(id -un)}" -f 'run_cpa3d.sh. lib' || true) | wc -l)"
    for k in 0 1 2 3 4; do
      t=0; for sn in yolov8m yolo11s rtdetr; do [ -f $OUT/fold$k/textures/npa_stop__$sn/stage_done.json ] && t=$((t+1)); done
      e1=$([ -f $OUT/fold$k/eval_npa_sweep/stage_done.json ] && echo yes || echo no)
      e2=$([ -f $OUT/fold$k/eval_transfer_nat/stage_done.json ] && echo yes || echo no)
      echo "fold $k: sweep eval $e1 | transfer patches $t/3  eval $e2"
    done
    tail -n 2 $OUT/logs/sweep_gpu*.log $OUT/logs/trnat_gpu*.log 2>/dev/null || true
    grep -h FAILED $OUT/logs/sweep_gpu*.log $OUT/logs/trnat_gpu*.log 2>/dev/null || true
    ;;
  sweep_trnat_report)
    $PY "$HERE/npa_naturalness.py" --runs "$OUT"
    $PY "$HERE/cpa_analyze.py" --runs "$OUT" --eval-dir eval_npa_sweep --baseline npa_stop20 --report-name npa_sweep_report
    $PY "$HERE/cpa_transfer.py" --runs "$OUT" --eval-dir eval_transfer_nat --report-name transfer_nat_report
    ;;
  diversity)
    echo "variants: $DIV_VARIANTS   GPUs: $GPUS"
    launch_spread div_chain diversity
    ;;
  diversity_status)
    echo "workers running: $( (pgrep -u "${USER:-$(id -un)}" -f 'run_cpa3d.sh. lib' || true) | wc -l)"
    for k in 0 1 2 3 4; do
      t=0; for v in $DIV_VARIANTS; do [ -f $OUT/fold$k/textures/$v/stage_done.json ] && t=$((t+1)); done
      e=$([ -f $OUT/fold$k/eval_diversity/stage_done.json ] && echo yes || echo no)
      echo "fold $k: diversity patches $t/4  eval $e"
    done
    tail -n 2 $OUT/logs/diversity_gpu*.log 2>/dev/null || true
    grep -h FAILED $OUT/logs/diversity_gpu*.log 2>/dev/null || true
    ;;
  diversity_report)
    $PY "$HERE/cpa_diversity.py" --runs "$OUT"
    ;;
  transfer)
    [ -f "$BIGGAN" ] || { echo "BigGAN weights not found: $BIGGAN"; exit 1; }
    echo "surrogates: yolov8s(main) $TR_SURR | attacks: $TR_ATTACKS | GPUs: $GPUS"
    launch_spread tr_chain transfer
    ;;
  transfer_status)
    echo "workers running: $( (pgrep -u "${USER:-$(id -un)}" -f 'run_cpa3d.sh. lib' || true) | wc -l)"
    for k in 0 1 2 3 4; do
      t=0; n=0
      for sn in $TR_SURR; do for a in $TR_ATTACKS; do n=$((n+1)); [ -f $OUT/fold$k/textures/${a}__$sn/stage_done.json ] && t=$((t+1)); done; done
      e=$([ -f $OUT/fold$k/eval_transfer/stage_done.json ] && echo yes || echo no)
      echo "fold $k: transfer patches $t/$n  eval $e"
    done
    tail -n 2 $OUT/logs/transfer_gpu*.log 2>/dev/null || true
    grep -h FAILED $OUT/logs/transfer_gpu*.log 2>/dev/null || true
    ;;
  transfer_report)
    $PY "$HERE/cpa_transfer.py" --runs "$OUT"
    ;;
  lib) ;; # sourced by the workers
  status)
    echo "workers running: $( (pgrep -u "${USER:-$(id -un)}" -f 'run_cpa3d.sh. lib' || true) | wc -l)"
    for k in 0 1 2 3 4; do
      t=$( (ls $OUT/fold$k/textures/*/stage_done.json 2>/dev/null || true) | wc -l)
      g=$( (ls $OUT/fold$k/targets/*/done 2>/dev/null || true) | wc -l)
      e=$([ -f $OUT/fold$k/eval/stage_done.json ] && echo yes || echo no)
      echo "fold $k: textures $t/4  targets $g/3  eval $e"
    done
    tail -n 2 $OUT/logs/worker_gpu*.log 2>/dev/null || true
    grep -h FAILED $OUT/logs/worker_gpu*.log 2>/dev/null || true
    ;;
  report)
    $PY "$HERE/cpa_analyze.py" --runs "$OUT"
    echo; echo "black-box detectors, held-out mAP50:"
    $PY - "$OUT" <<'PYEOF'
import json, sys, glob, statistics as st
from collections import defaultdict
v = defaultdict(list)
for f in sorted(glob.glob(f"{sys.argv[1]}/fold*/targets/*/metrics/val_metrics.json")):
    j = json.load(open(f)); t = f.split("/")[-3]
    v[t].append(j.get("overall", j).get("mAP50", float("nan")))
for t, xs in v.items():
    print(f"  {t:<8} mAP50 {st.mean(xs):.3f} +- {st.pstdev(xs):.3f}  ({len(xs)} folds)")
PYEOF
    ;;
  *) sed -n '2,20p' "$0" ;;
esac
