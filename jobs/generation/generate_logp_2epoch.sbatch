#!/bin/bash
#SBATCH --job-name=cfggen300
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/intern1/CFG_test/cfg/logs/generation/cfggen300_%j.out
#SBATCH --error=/home/intern1/CFG_test/cfg/logs/generation/cfggen300_%j.err

set -euo pipefail

cd /home/intern1/CFG_test/cfg

OUT="outputs/generation/logp_cfg_scale2_2epoch_100_each_chemfm_tokens.csv"

# Refuse to overwrite previous results.
if [ -e "$OUT" ]; then
    echo "ERROR: Output already exists: $OUT"
    exit 2
fi

# Check required files inside the batch job.
test -f configs/generation/logp.json
test -f configs/scaling/guacamol_50k.json
test -f data/guacamol/guacamol_50000_normalized/train_50000_refnorm.csv
test -f src/generation/chemfm_cfg_generate.py
test -f checkpoints/lora_cfg_50000_chemfm_2epoch/adapter/adapter_config.json
test -f checkpoints/lora_cfg_50000_chemfm_2epoch/tokenizer/tokenizer.json

echo "===== GENERATION JOB ====="
date
hostname
nvidia-smi

python -u -m src.generation.chemfm_cfg_generate \
  --generation-config configs/generation/logp.json \
  --base-model /home/intern1/demo/demo_model \
  --adapter-dir checkpoints/lora_cfg_50000_chemfm_2epoch/adapter \
  --tokenizer-dir checkpoints/lora_cfg_50000_chemfm_2epoch/tokenizer \
  --stats-file configs/scaling/guacamol_50k.json \
  --training-smiles-csv data/guacamol/guacamol_50000_normalized/train_50000_refnorm.csv \
  --samples-per-target 100 \
  --guidance-scale 2.0 \
  --top-p 1.0 \
  --seed 42 \
  --output-csv "$OUT"

echo "===== GENERATION FINISHED ====="
date
