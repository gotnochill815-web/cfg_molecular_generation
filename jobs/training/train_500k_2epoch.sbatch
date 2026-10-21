#!/bin/bash
#SBATCH --job-name=ws2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/intern1/CFG_test/cfg/logs/training/ws2_%j.out
#SBATCH --error=/home/intern1/CFG_test/cfg/logs/training/ws2_%j.err

set -euo pipefail
cd /home/intern1/CFG_test/cfg

TRAIN_OUT="checkpoints/lora_cfg_500000_chemfm_2epoch"

echo "===== 500K CFG TRAINING ====="
date
hostname
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Job name: ${SLURM_JOB_NAME:-unknown}"

# Protect existing checkpoints and validate required inputs.
if [ -e "$TRAIN_OUT" ]; then
    echo "ERROR: Checkpoint output already exists: $TRAIN_OUT"
    exit 2
fi

test -f configs/dataset/guacamol_500k_normalized.yaml
test -f configs/training/default_500k_chemfm_2epoch.yaml
test -f configs/model/olmo_7b.yaml
test -f train_full.py
test -f data/guacamol_500k/normalized/train_500k_refnorm.csv
test -f data/guacamol_500k/normalized/val_500k_refnorm.csv
test -f data/guacamol_500k/normalized/test_500k_refnorm.csv
test -d /home/intern1/demo/demo_model

echo "===== GPU STATUS ====="
nvidia-smi

echo "===== PYTHON AND CUDA CHECK ====="
which python
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA unavailable inside GPU job"'

echo "===== TRAINING CONFIGURATION ====="
cat configs/training/default_500k_chemfm_2epoch.yaml

echo "===== STARTING TRAINING ====="
python -u train_full.py \
  --dataset configs/dataset/guacamol_500k_normalized.yaml \
  --model configs/model/olmo_7b.yaml \
  --training configs/training/default_500k_chemfm_2epoch.yaml

echo "===== TRAINING COMPLETED ====="
date
