#!/usr/bin/env bash
# Detector variants on the same folds: input 1280 px (cv_s1280) and DOTA-initialised backbone (cv_dota1280).
#   bash improve.sh detectors | status | compare | patches <out> | report <out>
set -euo pipefail
cd "$(dirname "$0")"

ROOT="${ROOT:-../data/Aerial Image Dataset/yolo_dataset}"
COMMON="--epochs 15 --imgsz 1280 --batch 8 --patience 5 --freeze 10 --workers 4"
declare -A DET=(
  [cv_s1280]="--model yolov8s.pt $COMMON"
  [cv_dota1280]="--model yolov8s.yaml --init-weights yolov8s-obb.pt $COMMON"
)
PATCH_EPOCHS="${PATCH_EPOCHS:-50}" # the 80-epoch run plateaued from ~epoch 30
SEEDS="${SEEDS:-0,1}"
METHODS="eot cpa_light cpa_shadow cpa_ls"

launch() { # launch <gpu> <out> <folds> <logname> <extra cv_run args...>
  local g=$1 out=$2 folds=$3 name=$4; shift 4
  local cmd=(python cv_run.py --root "$ROOT" --out "$out" --folds "$folds" --det-args "${DET[$out]}" "$@")
  if [ "${DRY:-0}" = "1" ]; then echo "GPU $g: ${cmd[*]}"; return; fi
  mkdir -p improve_logs
  CUDA_VISIBLE_DEVICES=$g nohup "${cmd[@]}" > "improve_logs/${out}_${name}.log" 2>&1 &
  echo "started GPU $g  $out  folds $folds  $name  (log: improve_logs/${out}_${name}.log)"
}

case "${1:-}" in
  detectors)
    [ -f yolov8s-obb.pt ] || { echo "yolov8s-obb.pt missing"; exit 1; }
    [ -f yolov8s.pt ] || { echo "yolov8s.pt missing"; exit 1; }
    [ -f "$ROOT/cv/folds.json" ] || { echo "folds missing: $ROOT/cv/folds.json"; exit 1; }
    launch 0 cv_s1280    0,4 det_f0_4 --stages detector
    launch 1 cv_s1280    1   det_f1   --stages detector
    launch 2 cv_s1280    2   det_f2   --stages detector
    launch 3 cv_s1280    3   det_f3   --stages detector
    launch 0 cv_dota1280 0   det_f0   --stages detector
    launch 1 cv_dota1280 1,4 det_f1_4 --stages detector
    launch 2 cv_dota1280 2   det_f2   --stages detector
    launch 3 cv_dota1280 3   det_f3   --stages detector
    ;;
  patches)
    out="${2:?usage: bash improve.sh patches cv_dota1280|cv_s1280}"
    [ -n "${DET[$out]+x}" ] || { echo "unknown variant $out"; exit 1; }
    n=$( (ls "$out"/fold*/detector/stage_done.json 2>/dev/null || true) | wc -l)
    [ "$n" = "5" ] || { echo "$out has $n/5 finished detectors -- wait for phase 1 (bash improve.sh status)"; exit 1; }
    i=0
    for f in 0 1 2 3 4; do
      for m in $METHODS; do
        launch $((i % 4)) "$out" "$f" "f${f}_${m}" --stages patch,eval,mapdrop \
          --methods "$m" --seeds "$SEEDS" --patch-epochs "$PATCH_EPOCHS"
        i=$((i + 1))
      done
    done
    ;;
  status)
    echo "cv_run jobs running: $( (pgrep -u "${USER:-$(id -un)}" -f cv_run.py || true) | wc -l)"
    for o in cv_s1280 cv_dota1280; do
      d=$( (ls $o/fold*/detector/stage_done.json 2>/dev/null || true) | wc -l)
      e=$( (ls $o/fold*/eval/*/stage_done.json 2>/dev/null || true) | wc -l)
      echo "$o: $d/5 detectors done, $e patch evaluations done"
    done
    grep -l "FAILED" improve_logs/*.log 2>/dev/null | sed 's/^/FAILED in: /' || true
    ;;
  compare)
    for o in cv_runs_e80 cv_s1280 cv_dota1280; do
      echo; echo "=================== $o ==================="
      python cv_analyze.py --out "$o" | sed -n '/## 1\./,/## 2\./p' | grep -v "^## 2"
    done
    ;;
  report)
    out="${2:?usage: bash improve.sh report cv_dota1280}"
    python cv_analyze.py --out "$out"
    ;;
  *)
    sed -n '2,13p' "$0"; exit 1 ;;
esac
