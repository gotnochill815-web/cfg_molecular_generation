#!/usr/bin/env bash
set -euo pipefail

cd /home/intern1/projects/cfg

OUT="outputs/generation/guidance_sweep"
CONFIG="configs/generation/logp.json"

mkdir -p "$OUT"

echo "===== GPU CHECK ====="
hostname
nvidia-smi

echo
echo "===== SMOKE TEST ====="
echo "Testing one molecule for each LogP target at scale 2.0"

python -m src.generation.chemfm_cfg_generate \
  --generation-config "$CONFIG" \
  --samples-per-target 1 \
  --guidance-scale 2.0 \
  --seed 42 \
  --output-csv "$OUT/smoke_scale_2p0.csv" \
  2>&1 | tee "$OUT/smoke_scale_2p0.log"

echo
echo "Smoke test passed. Starting sweep..."

for SCALE in 1.0 1.5 2.0 2.5 3.0 3.5 4.0; do
    TAG="${SCALE//./p}"
    CSV="$OUT/logp_cfg_scale_${TAG}.csv"
    LOG="$OUT/logp_cfg_scale_${TAG}.log"

    echo
    echo "=================================================="
    echo "GUIDANCE SCALE: $SCALE"
    echo "Output: $CSV"
    echo "=================================================="

    python -m src.generation.chemfm_cfg_generate \
      --generation-config "$CONFIG" \
      --samples-per-target 10 \
      --guidance-scale "$SCALE" \
      --seed 42 \
      --output-csv "$CSV" \
      2>&1 | tee "$LOG"
done

echo
echo "All seven guidance-scale runs completed."

python - <<'PY'
import csv
import json
from pathlib import Path

out = Path("outputs/generation/guidance_sweep")
summaries = sorted(out.glob("logp_cfg_scale_*.summary.json"))

rows = []
for path in summaries:
    with path.open() as f:
        s = json.load(f)

    logp_stats = s.get("property_error_summary", {}).get("logp", {})

    rows.append({
        "guidance_scale": s.get("guidance_scale"),
        "requested": s.get("requested_generations"),
        "validity": s.get("validity"),
        "uniqueness_among_valid": s.get("unique_fraction_among_valid"),
        "novelty_among_valid": s.get("novelty_among_valid"),
        "logp_mae": logp_stats.get("mean_absolute_error"),
        "elapsed_seconds": s.get("elapsed_seconds"),
    })

rows.sort(key=lambda r: float(r["guidance_scale"]))

if len(rows) != 7:
    print(f"WARNING: Expected 7 summaries, found {len(rows)}.")
    print("Check the individual logs for failed runs.")

comparison = out / "guidance_scale_comparison.csv"
fields = [
    "guidance_scale", "requested", "validity",
    "uniqueness_among_valid", "novelty_among_valid",
    "logp_mae", "elapsed_seconds"
]

with comparison.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("\n===== GUIDANCE SCALE COMPARISON =====")
print(
    f"{'Scale':>6} {'Valid':>8} {'Unique':>8} "
    f"{'Novel':>8} {'LogP MAE':>10} {'Time(s)':>9}"
)

for r in rows:
    def fmt(value, digits=3):
        return "n/a" if value is None else f"{value:.{digits}f}"

    print(
        f"{float(r['guidance_scale']):6.1f} "
        f"{fmt(r['validity']):>8} "
        f"{fmt(r['uniqueness_among_valid']):>8} "
        f"{fmt(r['novelty_among_valid']):>8} "
        f"{fmt(r['logp_mae']):>10} "
        f"{fmt(r['elapsed_seconds'], 1):>9}"
    )

eligible = [
    r for r in rows
    if r["logp_mae"] is not None and r["validity"] is not None
]
if eligible:
    best = min(eligible, key=lambda r: r["logp_mae"])
    print(
        "\nLowest observed LogP MAE:",
        best["guidance_scale"],
        "| MAE:",
        best["logp_mae"],
    )
    print(
        "Review validity, uniqueness, and novelty before choosing "
        "the final scale."
    )

print("Comparison CSV:", comparison)
PY

echo
echo "Sweep finished. Results are in: $OUT"
