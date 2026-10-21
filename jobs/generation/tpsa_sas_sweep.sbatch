#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/intern1/CFG_test/cfg/logs/generation/sweep_%j.out
#SBATCH --error=/home/intern1/CFG_test/cfg/logs/generation/sweep_%j.err

set -euo pipefail
cd /home/intern1/CFG_test/cfg

CONFIG="configs/generation/multi_property/tpsa_sas_grid.json"
STATS="configs/scaling/guacamol_500k.json"
TRAIN="data/guacamol_500k/normalized/train_500k_refnorm.csv"
BASE="/home/intern1/demo/demo_model"
ADAPTER="checkpoints/lora_cfg_500000_chemfm_2epoch/adapter"
TOKENIZER="checkpoints/lora_cfg_500000_chemfm_2epoch/tokenizer"

for FILE in \
  "$CONFIG" "$STATS" "$TRAIN" \
  "$ADAPTER/adapter_config.json" \
  "$ADAPTER/adapter_model.safetensors" \
  "$TOKENIZER/tokenizer.json" \
  "$BASE/config.json"
do
  test -f "$FILE" || { echo "Missing required file: $FILE"; exit 2; }
done

python -c 'import torch; assert torch.cuda.is_available(); print("CUDA available")'
hostname
nvidia-smi

for SCALE in 1.0 1.5 2.0 2.5 3.0; do
  OUT="outputs/generation/tpsa_sas_cfg_scale_${SCALE}.csv"

  if [ -e "$OUT" ]; then
    echo "Output already exists: $OUT"
    exit 2
  fi

  echo "===== GUIDANCE SCALE $SCALE ====="
  date

  python -u -m src.generation.chemfm_cfg_generate \
    --generation-config "$CONFIG" \
    --samples-per-target 5 \
    --guidance-scale "$SCALE" \
    --top-p 1.0 \
    --source-max-len 512 \
    --target-max-len 512 \
    --seed 42 \
    --base-model "$BASE" \
    --adapter-dir "$ADAPTER" \
    --tokenizer-dir "$TOKENIZER" \
    --stats-file "$STATS" \
    --training-smiles-csv "$TRAIN" \
    --output-csv "$OUT"

  echo "===== SCALE $SCALE COMPLETE ====="
  date
done

echo "===== SWEEP COMPLETE ====="
date
