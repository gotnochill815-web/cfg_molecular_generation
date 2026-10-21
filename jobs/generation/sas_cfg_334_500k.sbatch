#!/bin/bash
#SBATCH --job-name=sas334
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/intern1/CFG_test/cfg/logs/generation/sas334_%j.out
#SBATCH --error=/home/intern1/CFG_test/cfg/logs/generation/sas334_%j.err

set -euo pipefail
cd /home/intern1/CFG_test/cfg

OUTDIR="outputs/generation/sas_cfg_334_500k_2epoch"
test ! -e "$OUTDIR"
test -f configs/generation/sas.json
test -f configs/scaling/guacamol_500k.json
test -f data/guacamol_500k/normalized/train_500k_refnorm.csv
test -f checkpoints/lora_cfg_500000_chemfm_2epoch/adapter/adapter_model.safetensors
test -f checkpoints/lora_cfg_500000_chemfm_2epoch/tokenizer/tokenizer.json
test -d /home/intern1/demo/demo_model

echo "===== FULL SAS CFG GENERATION ====="
date
nvidia-smi

python -u -m src.generation.chemfm_cfg_generate \
  --generation-config configs/generation/sas.json \
  --base-model /home/intern1/demo/demo_model \
  --adapter-dir checkpoints/lora_cfg_500000_chemfm_2epoch/adapter \
  --tokenizer-dir checkpoints/lora_cfg_500000_chemfm_2epoch/tokenizer \
  --stats-file configs/scaling/guacamol_500k.json \
  --training-smiles-csv data/guacamol_500k/normalized/train_500k_refnorm.csv \
  --guidance-scale 2.5 \
  --top-p 1.0 \
  --source-max-len 512 \
  --target-max-len 512 \
  --output-csv "$OUTDIR/molecules.csv"

echo "===== FULL SAS CFG GENERATION COMPLETED ====="
date
