#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/intern1/CFG_test/cfg/logs/generation/logp334_%j.out
#SBATCH --error=/home/intern1/CFG_test/cfg/logs/generation/logp334_%j.err

set -euo pipefail
cd /home/intern1/CFG_test/cfg

OUTDIR="outputs/generation/logp_cfg_334_500k_2epoch"

if [ -e "$OUTDIR" ]; then
    echo "ERROR: Output directory exists: $OUTDIR"
    exit 2
fi

test -f configs/generation/logp.json
test -f configs/scaling/guacamol_500k.json
test -f data/guacamol_500k/normalized/train_500k_refnorm.csv
test -f checkpoints/lora_cfg_500000_chemfm_2epoch/adapter/adapter_config.json
test -f checkpoints/lora_cfg_500000_chemfm_2epoch/adapter/adapter_model.safetensors
test -f checkpoints/lora_cfg_500000_chemfm_2epoch/tokenizer/tokenizer.json
test -d /home/intern1/demo/demo_model

echo "===== LOGP CFG GENERATION ====="
date
nvidia-smi

python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())'

python -u -m src.generation.chemfm_cfg_generate \
  --generation-config configs/generation/logp.json \
  --samples-per-target 334 \
  --guidance-scale 1.5 \
  --top-p 1.0 \
  --source-max-len 512 \
  --target-max-len 512 \
  --seed 42 \
  --base-model /home/intern1/demo/demo_model \
  --adapter-dir checkpoints/lora_cfg_500000_chemfm_2epoch/adapter \
  --tokenizer-dir checkpoints/lora_cfg_500000_chemfm_2epoch/tokenizer \
  --stats-file configs/scaling/guacamol_500k.json \
  --training-smiles-csv data/guacamol_500k/normalized/train_500k_refnorm.csv \
  --output-csv "$OUTDIR/molecules.csv"

echo "===== LOGP CFG GENERATION COMPLETED ====="
date
