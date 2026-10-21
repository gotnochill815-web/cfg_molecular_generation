#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/intern1/CFG_test/cfg/logs/generation/tpsa_chemfm_%j.out
#SBATCH --error=/home/intern1/CFG_test/cfg/logs/generation/tpsa_chemfm_%j.err

set -euo pipefail
cd /home/intern1/CFG_test/cfg

OUTDIR="outputs/generation/tpsa_guidance_sweep_2epoch_chemfm_tokens"

if [ -e "$OUTDIR" ]; then
    echo "ERROR: Output directory exists: $OUTDIR"
    exit 2
fi

test -f configs/generation/tpsa.json
test -f configs/scaling/guacamol_50k.json
test -f data/guacamol/guacamol_50000_normalized/train_50000_refnorm.csv
test -f src/generation/tpsa_cfg_sweep_chemfm.py
test -f checkpoints/lora_cfg_50000_chemfm_2epoch/adapter/adapter_config.json
test -f checkpoints/lora_cfg_50000_chemfm_2epoch/tokenizer/tokenizer.json

echo "===== TPSA CHEMFM-STYLE SWEEP ====="
date
hostname
nvidia-smi

python -u -m src.generation.tpsa_cfg_sweep_chemfm \
  --generation-config configs/generation/tpsa.json \
  --guidance-scales 1.0 1.5 2.0 2.5 3.0 3.5 4.0 \
  --samples-per-target 10 \
  --seed 42 \
  --temperature 1.0 \
  --top-p 1.0 \
  --source-max-len 512 \
  --target-max-len 512 \
  --output-dir "$OUTDIR" \
  --base-model /home/intern1/demo/demo_model \
  --adapter-dir checkpoints/lora_cfg_50000_chemfm_2epoch/adapter \
  --tokenizer-dir checkpoints/lora_cfg_50000_chemfm_2epoch/tokenizer \
  --stats-file configs/scaling/guacamol_50k.json \
  --training-smiles-csv data/guacamol/guacamol_50000_normalized/train_50000_refnorm.csv

echo "===== SWEEP FINISHED ====="
date
