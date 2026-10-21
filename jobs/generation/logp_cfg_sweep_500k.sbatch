#!/bin/bash
#SBATCH --job-name=logp500k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/intern1/CFG_test/cfg/logs/generation/logp500k_%j.out
#SBATCH --error=/home/intern1/CFG_test/cfg/logs/generation/logp500k_%j.err

set -euo pipefail
cd /home/intern1/CFG_test/cfg

OUTDIR="outputs/generation/logp_cfg_sweep_500k_2epoch"

echo "===== 500K LOGP CFG SWEEP ====="
date
hostname

if [ -e "$OUTDIR" ]; then
    echo "ERROR: Output directory already exists: $OUTDIR"
    echo "Choose a new output directory to preserve previous results."
    exit 2
fi

test -f src/generation/logp_cfg_sweep_500k.py
test -f configs/generation/logp.json
test -f configs/scaling/guacamol_500k.json
test -f data/guacamol_500k/normalized/train_500k_refnorm.csv
test -f checkpoints/lora_cfg_500000_chemfm_2epoch/adapter/adapter_config.json
test -f checkpoints/lora_cfg_500000_chemfm_2epoch/adapter/adapter_model.safetensors
test -f checkpoints/lora_cfg_500000_chemfm_2epoch/tokenizer/tokenizer.json
test -d /home/intern1/demo/demo_model

echo "===== GPU ====="
nvidia-smi

python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA unavailable"'

echo "===== STARTING CFG GUIDANCE SWEEP ====="
python -u -m src.generation.logp_cfg_sweep_500k \
  --generation-config configs/generation/logp.json \
  --guidance-scales 1.0 1.5 2.0 2.5 3.0 3.5 4.0 \
  --samples-per-target 10 \
  --seed 42 \
  --temperature 1.0 \
  --top-p 1.0 \
  --source-max-len 512 \
  --target-max-len 512 \
  --base-model /home/intern1/demo/demo_model \
  --adapter-dir checkpoints/lora_cfg_500000_chemfm_2epoch/adapter \
  --tokenizer-dir checkpoints/lora_cfg_500000_chemfm_2epoch/tokenizer \
  --stats-file configs/scaling/guacamol_500k.json \
  --training-smiles-csv data/guacamol_500k/normalized/train_500k_refnorm.csv \
  --output-dir "$OUTDIR"

echo "===== LOGP CFG SWEEP COMPLETED ====="
date
