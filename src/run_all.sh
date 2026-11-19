#!/usr/bin/env bash
# 2-D stage, all folds: photo-grouped folds -> detector -> EOT and CPA patches -> evaluation -> report.
#   GPU=0 bash run_all.sh            (SMOKE=1 for a quick end-to-end check; SEEDS=0,1,2 for 3 patch seeds)
# Env: ROOT (yolo_dataset), GPU, DET_ARGS, METHODS, SEEDS, OUT (default cv_runs), FOLDS. Finished stages are skipped.
set -euo pipefail
cd "$(dirname "$0")"

ROOT="${ROOT:-../data/Aerial Image Dataset/yolo_dataset}"
GPU="${GPU:?set GPU=<id>  (look at nvidia-smi first; this is a shared machine)}"
DET_ARGS="${DET_ARGS:---model yolov8s.pt --epochs 15 --imgsz 960 --batch 8 --patience 5 --workers 4}"
METHODS="${METHODS:-eot,cpa_light,cpa_shadow,cpa_ls}"
SEEDS="${SEEDS:-0}"
OUT="${OUT:-cv_runs}"
FOLDS="${FOLDS:-all}"
export CUDA_VISIBLE_DEVICES="$GPU"

for f in make_group_split.py cv_run.py cv_analyze.py patch_map_drop.py patch_evaluate.py patch_train.py \
         evaluate_yolov8.py train_yolov8.py dataset_utils.py photo_stats.py corruptions.py; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
python -m py_compile make_group_split.py cv_run.py cv_analyze.py patch_map_drop.py patch_evaluate.py \
  patch_train.py evaluate_yolov8.py train_yolov8.py dataset_utils.py photo_stats.py corruptions.py
python -c "import ultralytics, torch; print('ultralytics', ultralytics.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

MODEL=$(python -c "import shlex,sys; a=shlex.split(sys.argv[1]); print(a[a.index('--model')+1] if '--model' in a else 'yolov8s.pt')" "$DET_ARGS")
[ -f "$MODEL" ] || { echo "$MODEL not found"; exit 1; }

(
  flock 9
  if [ ! -f "$ROOT/cv/folds.json" ]; then
    echo "== building photo-grouped folds =="
    python make_group_split.py --root "$ROOT"
  fi
) 9>"$ROOT/.cv_split.lock"

if [ "${SMOKE:-0}" = "1" ]; then
  echo "== smoke test =="
  python cv_run.py --root "$ROOT" --out cv_smoke --folds 0 \
    --det-args "--model yolov8s.pt --epochs 1 --imgsz 640 --batch 8 --workers 2" \
    --methods eot,cpa_ls --patch-epochs 1 --instances-per-epoch 40 --max-files-per-photo 1
  python cv_analyze.py --out cv_smoke
  echo "cv_smoke/cv_report.md"
  exit 0
fi

echo "== full run: folds=$FOLDS methods=$METHODS seeds=$SEEDS =="
python cv_run.py --root "$ROOT" --out "$OUT" --folds "$FOLDS" \
  --det-args "$DET_ARGS" --methods "$METHODS" --seeds "$SEEDS"

if [ "$FOLDS" = "all" ]; then
  python cv_analyze.py --out "$OUT"
else
  echo "folds done: $FOLDS"
fi
