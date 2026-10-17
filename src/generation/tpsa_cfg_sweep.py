"""Sweep CFG guidance scales while loading the model only once."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch

from src.cfg.decoder import cfg_decode
from src.generation.prompt_builder import build_prompt
from src.utils.property_scaling import normalize_targets
from src.generation.chemfm_cfg_generate import (
    ROOT,
    PROPERTIES,
    load_model,
    load_training_smiles,
    measure_properties,
    validate_condition_config,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--generation-config",
        required=True,
    )
    parser.add_argument(
        "--guidance-scales",
        nargs="+",
        type=float,
        default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    )
    parser.add_argument("--samples-per-target", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        default="outputs/generation/guidance_sweep",
    )
    parser.add_argument(
        "--base-model",
        default="/path/to/your/base-model",
    )
    parser.add_argument(
        "--adapter-dir",
        default=str(
            ROOT / "checkpoints/lora_cfg_50000_chemfm_1epoch/adapter"
        ),
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=str(
            ROOT / "checkpoints/lora_cfg_50000_chemfm_1epoch/tokenizer"
        ),
    )

    args = parser.parse_args()

    if args.samples_per_target < 1:
        parser.error("--samples-per-target must be positive")
    if not args.guidance_scales:
        parser.error("At least one guidance scale is required")
    if any(s < 0 for s in args.guidance_scales):
        parser.error("Guidance scales must be nonnegative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")

    config_path = Path(args.generation_config)
    with config_path.open() as f:
        config = json.load(f)

    # Validate every condition before loading the large model.
    condition_groups = []
    for key, spec in config["properties"].items():
        names = validate_condition_config(key, spec)
        for point_index, point in enumerate(spec["sample_points"]):
            values = [point] if len(names) == 1 and not isinstance(
                point, (list, tuple)
            ) else list(point)

            if len(values) != len(names):
                raise ValueError(
                    f"Condition {key}: {point!r} does not match {names}"
                )

            raw_targets = {
                name: float(value)
                for name, value in zip(names, values)
            }
            condition_groups.append(
                (key, spec, point_index, raw_targets)
            )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CFG GUIDANCE SWEEP")
    print("Scales:", args.guidance_scales)
    print("Samples per target:", args.samples_per_target)
    print("Loading base model and adapter ONCE...")
    print("=" * 70)

    # The one and only model load for the whole sweep.
    model, tokenizer = load_model(
        args.base_model,
        args.adapter_dir,
        args.tokenizer_dir,
    )

    train_smiles = load_training_smiles()
    unconditional_prompt = "<NULL>\n<molstart>"
    comparisons = []

    print("\nMODEL LOADED. Reusing it for every guidance scale.")

    for scale in args.guidance_scales:
        # Use a repeatable seed per scale for a more comparable experiment.
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        started = time.time()
        records = []

        print("\n" + "=" * 70)
        print(f"GUIDANCE SCALE: {scale}")
        print("=" * 70)

        for key, spec, point_index, raw_targets in condition_groups:
            normalized = normalize_targets(**raw_targets)
            conditional_prompt = build_prompt(**normalized)

            temperature = (
                args.temperature
                if args.temperature is not None
                else float(spec.get("temperature", 1.0))
            )

            print(
                f"Condition={key}, point={point_index}, "
                f"raw={raw_targets}, normalized={normalized}"
            )

            for sample_index in range(args.samples_per_target):
                generated = cfg_decode(
                    model=model,
                    tokenizer=tokenizer,
                    conditional_prompt=conditional_prompt,
                    unconditional_prompt=unconditional_prompt,
                    guidance_scale=scale,
                    temperature=temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    stop_newline=True,
                    verbose=False,
                )

                smiles, measured = measure_properties(generated)
                valid = bool(smiles)
                novel = bool(valid and smiles not in train_smiles)

                row = {
                    "guidance_scale": scale,
                    "condition_name": key,
                    "condition_index": point_index,
                    "sample_index": sample_index,
                    "smiles": smiles,
                    "valid": valid,
                    "novel": novel if valid else None,
                    "temperature": temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "conditional_prompt": conditional_prompt,
                    "unconditional_prompt": unconditional_prompt,
                }

                for name in PROPERTIES:
                    target = raw_targets.get(name)
                    row[f"{name}_target_raw"] = target
                    row[f"{name}_target_normalized"] = normalized.get(name)
                    row[f"{name}_measured"] = measured.get(name)
                    row[f"{name}_absolute_error"] = (
                        abs(measured[name] - target)
                        if target is not None and measured.get(name) is not None
                        else None
                    )

                records.append(row)

        elapsed = time.time() - started
        valid_rows = [r for r in records if r["valid"]]
        unique_smiles = {r["smiles"] for r in valid_rows}
        novel_count = sum(bool(r["novel"]) for r in valid_rows)

        error_summary = {}
        for name in PROPERTIES:
            errors = [
                r[f"{name}_absolute_error"]
                for r in records
                if r[f"{name}_absolute_error"] is not None
            ]
            if errors:
                error_summary[name] = {
                    "n": len(errors),
                    "mean_absolute_error": sum(errors) / len(errors),
                }

        tag = f"{scale:.1f}".replace(".", "p")
        csv_path = output_dir / f"tpsa_cfg_scale_{tag}.csv"
        summary_path = output_dir / f"tpsa_cfg_scale_{tag}.summary.json"

        if records:
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(records[0].keys())
                )
                writer.writeheader()
                writer.writerows(records)

        summary = {
            "guidance_scale": scale,
            "requested_generations": len(records),
            "valid_generations": len(valid_rows),
            "validity": (
                len(valid_rows) / len(records) if records else 0.0
            ),
            "unique_valid_smiles": len(unique_smiles),
            "unique_fraction_among_valid": (
                len(unique_smiles) / len(valid_rows)
                if valid_rows else 0.0
            ),
            "novel_valid_generations": novel_count,
            "novelty_among_valid": (
                novel_count / len(valid_rows) if valid_rows else 0.0
            ),
            "property_error_summary": error_summary,
            "elapsed_seconds": round(elapsed, 2),
            "output_csv": str(csv_path),
        }

        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        tpsa_error = error_summary.get(
            "tpsa", {}).get(
            "mean_absolute_error"
        )

        comparisons.append({
            "guidance_scale": scale,
            "requested": len(records),
            "validity": summary["validity"],
            "uniqueness": summary["unique_fraction_among_valid"],
            "novelty": summary["novelty_among_valid"],
            "tpsa_mae": tpsa_error,
            "elapsed_seconds": round(elapsed, 2),
        })

        print(
            f"Scale {scale}: valid={summary['validity']:.3f}, "
            f"unique={summary['unique_fraction_among_valid']:.3f}, "
            f"novel={summary['novelty_among_valid']:.3f}, "
            f"TPSA MAE={tpsa_error}, time={elapsed:.1f}s"
        )

        del records
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # IMPORTANT: model stays loaded.
        print("Reusing the same loaded model for the next scale.")

    comparison_path = output_dir / "guidance_scale_comparison.csv"
    with comparison_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "guidance_scale",
                "requested",
                "validity",
                "uniqueness",
                "novelty",
                "tpsa_mae",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(comparisons)

    print("\n===== SWEEP COMPLETE =====")
    print("Comparison:", comparison_path)
    for r in comparisons:
        print(r)


if __name__ == "__main__":
    main()
