"""SAS + LogP classifier-free guidance sweep for the 500k CFG adapter."""

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path

import torch

from src.cfg.decoder import cfg_decode
from src.generation.prompt_builder import build_prompt
from src.generation.chemfm_cfg_generate import (
    ROOT,
    PROPERTIES,
    load_model,
    load_property_stats,
    load_training_smiles,
    measure_properties,
    normalize_targets_with_stats,
    validate_condition_config,
)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep CFG guidance scales for joint SAS and LogP generation."
    )
    parser.add_argument(
        "--generation-config",
        default="configs/generation/multi_property/sas_logp_grid_500k_sweep.json",
    )
    parser.add_argument(
        "--guidance-scales",
        nargs="+",
        type=float,
        default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    )
    parser.add_argument("--samples-per-target", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--source-max-len", type=int, default=512)
    parser.add_argument("--target-max-len", type=int, default=512)

    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--stats-file", required=True)
    parser.add_argument("--training-smiles-csv", required=True)

    parser.add_argument(
        "--output-dir",
        default="outputs/generation/sas_logp_cfg_sweep_500k_2epoch_1to5",
    )

    args = parser.parse_args()

    if args.samples_per_target < 1:
        parser.error("--samples-per-target must be positive")
    if not args.guidance_scales or any(x < 0 for x in args.guidance_scales):
        parser.error("Guidance scales must be nonempty and nonnegative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.temperature is not None and args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.source_max_len < 1 or args.target_max_len < 1:
        parser.error("Source and target lengths must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    config_path = Path(args.generation_config).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open() as handle:
        config = json.load(handle)

    stats_path = Path(args.stats_file).expanduser()
    if not stats_path.is_absolute():
        stats_path = ROOT / stats_path
    stats = load_property_stats(stats_path)

    if "sas+logp" not in config.get("properties", {}):
        parser.error(
            "Generation config must contain a 'sas+logp' property entry"
        )

    condition_groups = []
    for key, spec in config["properties"].items():
        names = validate_condition_config(key, spec, stats)
        if names != ["sas", "logp"]:
            parser.error(
                f"This sweep expects SAS+LogP conditions, got {names}"
            )

        for index, point in enumerate(spec["sample_points"]):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                parser.error(
                    f"Expected each target as [SAS, LogP], got {point}"
                )
            raw_targets = {
                "sas": float(point[0]),
                "logp": float(point[1]),
            }
            condition_groups.append((key, spec, index, raw_targets))

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Choose a new output directory; existing results were not overwritten."
        )

    # Validate all file paths before loading model/GPU memory.
    required_paths = {
        "base model": Path(args.base_model).expanduser(),
        "adapter": Path(args.adapter_dir).expanduser(),
        "tokenizer": Path(args.tokenizer_dir).expanduser(),
        "training CSV": Path(args.training_smiles_csv).expanduser(),
    }
    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not (required_paths["adapter"] / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Missing adapter_config.json in {required_paths['adapter']}"
        )

    print("=" * 72)
    print("500K SAS + LOGP CFG GUIDANCE SWEEP")
    print("Scales:", args.guidance_scales)
    print(
        "Target pairs (SAS, LogP):",
        [(g[3]["sas"], g[3]["logp"]) for g in condition_groups],
    )
    print("Samples per target per scale:", args.samples_per_target)
    print("Expected generations:", len(args.guidance_scales)
          * len(condition_groups) * args.samples_per_target)
    print("Normalization statistics:", stats_path)
    print("Output directory:", output_dir)
    print("CFG guidance default in the shared runner: 2.0")
    print("=" * 72)

    # Load the base model and adapter only once for the entire sweep.
    model, tokenizer = load_model(
        str(required_paths["base model"]),
        str(required_paths["adapter"]),
        str(required_paths["tokenizer"]),
    )
    train_smiles = load_training_smiles(required_paths["training CSV"])

    unconditional_prompt = "<NULL>\n<molstart>"
    summaries = []
    output_dir.mkdir(parents=True, exist_ok=False)

    total = len(args.guidance_scales) * len(condition_groups) * args.samples_per_target
    overall_finished = 0

    for scale in args.guidance_scales:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        records = []
        started = time.time()
        print(f"\n===== GUIDANCE SCALE {scale} =====")

        for condition_name, spec, condition_index, raw_targets in condition_groups:
            normalized = normalize_targets_with_stats(raw_targets, stats)
            conditional_prompt = build_prompt(**normalized)
            temperature = (
                args.temperature
                if args.temperature is not None
                else float(spec.get("temperature", 1.0))
            )

            prompt_length = tokenizer(
                conditional_prompt,
                return_tensors="pt",
            )["input_ids"].shape[1]
            max_new_tokens = (
                args.max_new_tokens
                if args.max_new_tokens is not None
                else args.source_max_len + args.target_max_len - prompt_length
            )
            if max_new_tokens < 1:
                raise ValueError(
                    f"Invalid token budget {max_new_tokens} for {conditional_prompt!r}"
                )

            print(
                f"Target SAS={raw_targets['sas']} "
                f"(normalized={normalized['sas']:.4f}), "
                f"LogP={raw_targets['logp']} "
                f"(normalized={normalized['logp']:.4f}), scale={scale}"
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
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=0,
                    # Use the project's ChemFM-style length-budget settings.
                    stop_newline=False,
                    stop_separators=False,
                    block_repeats=False,
                    balance_parens=False,
                    verbose=False,
                )
                smiles, measured = measure_properties(generated)
                valid = bool(smiles)
                novel = bool(valid and smiles not in train_smiles)
                actual_sas = measured.get("sas")
                actual_logp = measured.get("logp")

                row = {
                    "guidance_scale": scale,
                    "condition_name": condition_name,
                    "condition_index": condition_index,
                    "sample_index": sample_index,
                    "target_sas": raw_targets["sas"],
                    "target_sas_normalized": normalized["sas"],
                    "measured_sas": actual_sas,
                    "sas_absolute_error": (
                        abs(actual_sas - raw_targets["sas"])
                        if actual_sas is not None else None
                    ),
                    "target_logp": raw_targets["logp"],
                    "target_logp_normalized": normalized["logp"],
                    "measured_logp": actual_logp,
                    "logp_absolute_error": (
                        abs(actual_logp - raw_targets["logp"])
                        if actual_logp is not None else None
                    ),
                    "smiles": smiles,
                    "valid": valid,
                    "novel": novel if valid else None,
                    "temperature": temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": max_new_tokens,
                    "conditional_prompt": conditional_prompt,
                    "unconditional_prompt": unconditional_prompt,
                }
                for name in PROPERTIES:
                    row[f"{name}_measured"] = measured.get(name)
                records.append(row)
                overall_finished += 1

                print(
                    f"[{overall_finished}/{total}] scale={scale} "
                    f"valid={valid} novel={novel if valid else 'n/a'} "
                    f"SAS={actual_sas} LogP={actual_logp} smiles={smiles}"
                )

        valid_rows = [row for row in records if row["valid"]]
        unique_smiles = {row["smiles"] for row in valid_rows}
        novel_rows = [row for row in valid_rows if row["novel"]]
        sas_errors = [
            row["sas_absolute_error"]
            for row in records
            if row["sas_absolute_error"] is not None
        ]
        logp_errors = [
            row["logp_absolute_error"]
            for row in records
            if row["logp_absolute_error"] is not None
        ]

        sas_mad = (
            sum(sas_errors) / len(sas_errors) if sas_errors else None
        )
        logp_mad = (
            sum(logp_errors) / len(logp_errors) if logp_errors else None
        )
        sas_absolute_deviation_std = (
            statistics.stdev(sas_errors) if len(sas_errors) > 1 else None
        )
        logp_absolute_deviation_std = (
            statistics.stdev(logp_errors) if len(logp_errors) > 1 else None
        )

        tag = str(scale).replace(".", "p")
        csv_path = output_dir / f"sas_logp_cfg_scale_{tag}.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        summary = {
            "guidance_scale": scale,
            "generation_config": str(config_path),
            "statistics_file": str(stats_path),
            "base_model": str(required_paths["base model"]),
            "adapter_dir": str(required_paths["adapter"]),
            "tokenizer_dir": str(required_paths["tokenizer"]),
            "training_smiles_csv": str(required_paths["training CSV"]),
            "requested_generations": len(records),
            "valid_generations": len(valid_rows),
            "validity": len(valid_rows) / len(records) if records else 0.0,
            "unique_valid_smiles": len(unique_smiles),
            "unique_fraction_among_valid": (
                len(unique_smiles) / len(valid_rows) if valid_rows else 0.0
            ),
            "novel_valid_generations": len(novel_rows),
            "novelty_among_valid": (
                len(novel_rows) / len(valid_rows) if valid_rows else 0.0
            ),
            "sas_mad": sas_mad,
            "sas_absolute_deviation_std": sas_absolute_deviation_std,
            "logp_mad": logp_mad,
            "logp_absolute_deviation_std": logp_absolute_deviation_std,
            "elapsed_seconds": round(time.time() - started, 2),
            "csv_path": str(csv_path),
        }
        (output_dir / f"sas_logp_cfg_scale_{tag}.summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        summaries.append(summary)
        print("Scale summary:", json.dumps(summary, indent=2))

    summary_path = output_dir / "summary_by_scale.csv"
    with summary_path.open("w", newline="") as handle:
        fields = [
            "guidance_scale", "requested_generations", "valid_generations",
            "validity", "unique_valid_smiles", "unique_fraction_among_valid",
            "novel_valid_generations", "novelty_among_valid",
            "sas_mad", "sas_absolute_deviation_std",
            "logp_mad", "logp_absolute_deviation_std",
            "elapsed_seconds", "csv_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field) for field in fields})

    # Also aggregate metrics for each target pair at each guidance scale.
    target_groups = {}
    for csv_file in sorted(output_dir.glob("sas_logp_cfg_scale_*.csv")):
        with csv_file.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    float(row["guidance_scale"]),
                    float(row["target_sas"]),
                    float(row["target_logp"]),
                )
                target_groups.setdefault(key, []).append(row)

    target_summary_path = output_dir / "summary_by_target_and_scale.csv"
    target_fields = [
        "guidance_scale", "target_sas", "target_logp",
        "requested_generations", "valid_generations", "validity",
        "unique_valid_smiles", "unique_fraction_among_valid",
        "novel_valid_generations", "novelty_among_valid",
        "sas_mad", "logp_mad",
    ]

    with target_summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=target_fields)
        writer.writeheader()

        for (scale, target_sas, target_logp), group in sorted(target_groups.items()):
            valid_rows = [
                row for row in group
                if row.get("valid", "").lower() == "true"
            ]
            unique_smiles = {row["smiles"] for row in valid_rows}
            novel_rows = [
                row for row in valid_rows
                if row.get("novel", "").lower() == "true"
            ]

            def numeric_errors(field):
                return [
                    float(row[field]) for row in group
                    if row.get(field) not in (None, "")
                ]

            sas_errors = numeric_errors("sas_absolute_error")
            logp_errors = numeric_errors("logp_absolute_error")

            writer.writerow({
                "guidance_scale": scale,
                "target_sas": target_sas,
                "target_logp": target_logp,
                "requested_generations": len(group),
                "valid_generations": len(valid_rows),
                "validity": len(valid_rows) / len(group) if group else 0.0,
                "unique_valid_smiles": len(unique_smiles),
                "unique_fraction_among_valid": (
                    len(unique_smiles) / len(valid_rows) if valid_rows else 0.0
                ),
                "novel_valid_generations": len(novel_rows),
                "novelty_among_valid": (
                    len(novel_rows) / len(valid_rows) if valid_rows else 0.0
                ),
                "sas_mad": (
                    sum(sas_errors) / len(sas_errors) if sas_errors else None
                ),
                "logp_mad": (
                    sum(logp_errors) / len(logp_errors) if logp_errors else None
                ),
            })

    print("\nSweep finished.")
    print("Summary CSV:", summary_path)
    print("Target-pair summary CSV:", target_summary_path)
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
