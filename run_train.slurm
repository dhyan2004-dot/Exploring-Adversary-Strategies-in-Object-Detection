#!/bin/bash
#SBATCH --job-name=yolov8_baseline
#SBATCH --partition=gpu                # EDIT: your cluster's GPU partition name
#SBATCH --gres=gpu:v100:1              # EDIT: syntax varies by cluster (e.g. --gres=gpu:1)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00                # EDIT: walltime limit, format HH:MM:SS or D-HH:MM:SS
#SBATCH --output=yolov8_baseline_%j.log
#SBATCH --error=yolov8_baseline_%j.log

# ---- EDIT this section for your cluster ----
# Common patterns -- uncomment/adjust whichever applies:

# module load cuda/12.1 python/3.11          # if your cluster uses environment modules
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate yolov8                       # if using a conda env
# source ~/venvs/yolov8/bin/activate           # if using a plain venv instead

# ---------------------------------------------

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd "$SLURM_SUBMIT_DIR"   # or hardcode the path to where these scripts live, e.g.:
# cd /scratch/$USER/AE5510/YOLOv8

python prepare_dataset.py     # safe to re-run every time -- it rebuilds yolo_dataset from scratch
python train_yolov8.py \
    --model yolov8s.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 32 \
    --workers 8 \
    --cache ram \
    --device 0

echo "Job finished at $(date)"
