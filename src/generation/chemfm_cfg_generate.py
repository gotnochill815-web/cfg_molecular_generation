"""ChemFM-style GuacaMol generation using the project's CFG decoder."""

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import QED, Crippen, rdMolDescriptors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cfg.decoder import cfg_decode
from src.generation.prompt_builder import build_prompt
from src.utils.property_scaling import (
    PROPERTIES,
    load_property_stats,
    normalize_targets as normalize_targets_with_stats,
    verify_stats_match_dataset,
)

RDLogger.DisableLog("rdApp.*")

# RDKit's synthetic accessibility score is provided by its SA_Score
# contrib module on most installations.
SASCORER = None
try:
    sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sa_dir not in sys.path:
        sys.path.insert(0, sa_dir)
    import sascorer
    SASCORER = sascorer
except Exception as exc:
    print(f"Warning: SAS calculation unavailable: {exc}")

def canonical_smiles(smiles):
    if not smiles:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def measure_properties(smiles):
    """Return measured raw properties, or None when unavailable."""
    canonical = canonical_smiles(smiles)
    if not canonical:
        return canonical, {
            "qed": None, "logp": None, "tpsa": None, "sas": None
        }

    mol = Chem.MolFromSmiles(canonical)

    measured = {
        "qed": float(QED.qed(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "sas": (
            float(SASCORER.calculateScore(mol))
            if SASCORER is not None else None
        ),
    }
    return canonical, measured


def load_model(base_model_path, adapter_path, tokenizer_path):
    """Load the trained adapter with its matching tokenizer and base model."""
    base_path = Path(base_model_path).expanduser()
    adapter_path = Path(adapter_path).expanduser()
    tokenizer_path = Path(tokenizer_path).expanduser()

    for path in (base_path, adapter_path, tokenizer_path):
        if not path.exists():
            raise FileNotFoundError(f"Required model path not found: {path}")

    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(f"Missing {adapter_config_path}")

    with adapter_config_path.open() as handle:
        adapter_config = json.load(handle)

    recorded_base = adapter_config.get("base_model_name_or_path")
    print("Base model requested:", base_path)
    print("Adapter-recorded base:", recorded_base)
    print("Adapter path:", adapter_path)
    print("Tokenizer path:", tokenizer_path)

    if recorded_base and os.path.normpath(str(recorded_base)) != os.path.normpath(str(base_path)):
        print(
            "WARNING: The adapter config records a different base-model "
            "path/name. Confirm that the requested base is the exact model "
            "used during training."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    embedding_count = model.get_input_embeddings().weight.shape[0]
    if embedding_count != len(tokenizer):
        print(
            f"Resizing embeddings from {embedding_count} "
            f"to tokenizer size {len(tokenizer)}"
        )
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        is_trainable=False,
    )
    model.eval()

    return model, tokenizer


def load_training_smiles(csv_path):
    """Load canonical SMILES from the selected training CSV."""
    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {path}")

    result = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            canonical = canonical_smiles(row.get("smiles", ""))
            if canonical:
                result.add(canonical)

    print("Canonical training molecules:", len(result))
    return result


def validate_condition_config(property_key, spec, property_stats):
    names = [item.strip().lower() for item in property_key.split("+")]

    if not names or any(name not in PROPERTIES for name in names):
        raise ValueError(
            f"Unsupported property combination {property_key!r}. "
            f"Allowed properties: {PROPERTIES}"
        )

    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate properties in {property_key!r}")

    points = spec.get("sample_points")
    if not isinstance(points, list) or not points:
        raise ValueError(f"{property_key}: sample_points must be nonempty")

    if int(spec.get("num_samples", 0)) < 1:
        raise ValueError(f"{property_key}: num_samples must be positive")

    means = spec.get("means")
    stds = spec.get("stds")

    if means is not None and len(means) != len(names):
        raise ValueError(f"{property_key}: means length does not match properties")
    if stds is not None and len(stds) != len(names):
        raise ValueError(f"{property_key}: stds length does not match properties")

    # Stop early if a generation config carries statistics from another dataset.
    for i, name in enumerate(names):
        if means is not None and stds is not None:
            mean, std = property_stats[name]
            if not math.isclose(float(means[i]), mean, rel_tol=0, abs_tol=1e-4):
                raise ValueError(
                    f"{property_key}: configured mean for {name} is "
                    f"{means[i]}, but this model was trained with {mean}"
                )
            if not math.isclose(float(stds[i]), std, rel_tol=0, abs_tol=1e-4):
                raise ValueError(
                    f"{property_key}: configured std for {name} is "
                    f"{stds[i]}, but this model was trained with {std}"
                )

    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation-config",
        required=True,
        help="ChemFM-style JSON file describing raw target values.",
    )
    parser.add_argument(
        "--adapter-dir",
        required=True,
        help="Adapter directory matching the selected base model and tokenizer.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        required=True,
        help="Tokenizer directory matching the selected trained adapter.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model used to train the adapter.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
    )
    parser.add_argument(
        "--stats-file",
        required=True,
        help="Normalization statistics used during training.",
    )
    parser.add_argument(
        "--training-smiles-csv",
        required=True,
        help="Training SMILES CSV used to calculate novelty.",
    )
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="Defaults: LogP=1.5, TPSA=2.0, QED=2.5, SAS=2.5; explicit values override.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override maximum generated tokens; default follows ChemFM's total-length budget.",
    )
    parser.add_argument("--source-max-len", type=int, default=512)
    parser.add_argument("--target-max-len", type=int, default=512)
    parser.add_argument(
        "--samples-per-target",
        type=int,
        default=None,
        help="Override num_samples in the JSON config for a quick test.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.base_model:
        parser.error("Pass --base-model with the base model path")

    if args.guidance_scale is not None and args.guidance_scale < 0:
        raise ValueError("--guidance-scale must be >= 0")
    if not (0 < args.top_p <= 1):
        raise ValueError("--top-p must be in (0, 1]")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_path = Path(args.generation_config)
    with config_path.open() as handle:
        config = json.load(handle)

    property_stats = load_property_stats(args.stats_file)
    verify_stats_match_dataset(property_stats, args.training_smiles_csv)
    properties_config = config.get("properties")
    if not isinstance(properties_config, dict) or not properties_config:
        raise ValueError("Generation config must contain a nonempty 'properties' object.")

    if args.guidance_scale is None:
        configured_properties = set(properties_config)
        property_defaults = {
            "logp": 1.5,
            "tpsa": 2.0,
            "qed": 2.5,
            "sas": 2.5,
        }
        if len(configured_properties) == 1:
            only_property = next(iter(configured_properties))
            args.guidance_scale = property_defaults.get(only_property, 2.0)
        else:
            # Mixed-property generation retains the generic default.
            args.guidance_scale = 2.0

    # Validate all config entries before allocating model memory.
    validated = []
    for key, spec in properties_config.items():
        names = validate_condition_config(key, spec, property_stats)
        validated.append((key, spec, names))

    model, tokenizer = load_model(
        args.base_model,
        args.adapter_dir,
        args.tokenizer_dir,
    )

    unconditional_prompt = "<NULL>\n<molstart>"
    train_smiles = load_training_smiles(args.training_smiles_csv)

    records = []
    total = sum(
        len(spec["sample_points"]) *
        (args.samples_per_target or int(spec["num_samples"]))
        for _, spec, _ in validated
    )
    finished = 0
    started = time.time()

    print("=" * 72)
    print("ChemFM-style GuacaMol generation with classifier-free guidance")
    print("CFG equation preserved in src/cfg/guidance.py")
    print("Unconditional prompt:", repr(unconditional_prompt))
    print("Guidance scale:", args.guidance_scale)
    print("Total requested generations:", total)
    print("=" * 72)

    for condition_name, spec, names in validated:
        temperature = float(spec.get("temperature", 1.0))
        sample_count = (
            args.samples_per_target
            if args.samples_per_target is not None
            else int(spec["num_samples"])
        )

        for point_index, point in enumerate(spec["sample_points"]):
            if len(names) == 1 and not isinstance(point, (list, tuple)):
                values = [point]
            elif isinstance(point, (list, tuple)):
                values = list(point)
            else:
                raise ValueError(
                    f"{condition_name}: each multi-property point must be a list"
                )

            if len(values) != len(names):
                raise ValueError(
                    f"{condition_name}: point {point!r} does not match "
                    f"properties {names}"
                )

            raw_targets = {
                name: float(value)
                for name, value in zip(names, values)
            }

            # Scale requested raw targets with this model's training statistics.
            normalized_targets = normalize_targets_with_stats(raw_targets, property_stats)
            conditional_prompt = build_prompt(**normalized_targets)

            print(
                f"\nCondition {condition_name} #{point_index + 1}: "
                f"raw={raw_targets}, normalized={normalized_targets}"
            )
            print("Conditional prompt:", repr(conditional_prompt))

            for sample_index in range(sample_count):
                # Match ChemFM evaluation:
                # max_length = source_max_len + target_max_len
                # steps = max_length - input_length
                prompt_length = tokenizer(
                    conditional_prompt,
                    return_tensors="pt",
                )["input_ids"].shape[1]

                if args.max_new_tokens is None:
                    effective_max_new_tokens = (
                        args.source_max_len
                        + args.target_max_len
                        - prompt_length
                    )
                else:
                    effective_max_new_tokens = args.max_new_tokens

                if effective_max_new_tokens < 1:
                    raise ValueError(
                        f"Invalid generation token budget: "
                        f"{effective_max_new_tokens}"
                    )

                generated = cfg_decode(
                    model=model,
                    tokenizer=tokenizer,
                    conditional_prompt=conditional_prompt,
                    unconditional_prompt=unconditional_prompt,
                    guidance_scale=args.guidance_scale,
                    temperature=temperature,
                    top_p=args.top_p,
                    max_new_tokens=effective_max_new_tokens,
                    min_new_tokens=0,
                    stop_newline=False,
                    stop_separators=False,
                    block_repeats=False,
                    balance_parens=False,
                    verbose=args.verbose,
                )

                smiles, measured = measure_properties(generated.replace(' ', ''))
                valid = bool(smiles)
                novel = bool(valid and smiles not in train_smiles)

                row = {
                    "condition_name": condition_name,
                    "condition_index": point_index,
                    "sample_index": sample_index,
                    "smiles": smiles,
                    "valid": valid,
                    "novel": novel if valid else None,
                    "guidance_scale": args.guidance_scale,
                    "temperature": temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": effective_max_new_tokens,
                    "conditional_prompt": conditional_prompt,
                    "unconditional_prompt": unconditional_prompt,
                }

                for name in PROPERTIES:
                    raw = raw_targets.get(name)
                    normalized = normalized_targets.get(name)
                    actual = measured.get(name)

                    row[f"{name}_target_raw"] = raw
                    row[f"{name}_target_normalized"] = normalized
                    row[f"{name}_measured"] = actual
                    row[f"{name}_absolute_error"] = (
                        abs(actual - raw)
                        if actual is not None and raw is not None
                        else None
                    )

                records.append(row)
                finished += 1
                print(
                    f"[{finished}/{total}] valid={valid} "
                    f"novel={novel if valid else 'n/a'} smiles={smiles}"
                )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = list(records[0].keys()) if records else []
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    valid_rows = [row for row in records if row["valid"]]
    unique_smiles = {row["smiles"] for row in valid_rows}
    novel_unique_smiles = {
        row["smiles"] for row in valid_rows if row["novel"]
    }

    summary = {
        "generation_config": str(config_path),
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "tokenizer_dir": args.tokenizer_dir,
        "guidance_scale": args.guidance_scale,
        "requested_generations": len(records),
        "valid_generations": len(valid_rows),
        "validity": len(valid_rows) / len(records) if records else 0.0,
        "unique_valid_smiles": len(unique_smiles),
        "unique_fraction_among_valid": (
            len(unique_smiles) / len(valid_rows) if valid_rows else 0.0
        ),
        "novel_unique_smiles": len(novel_unique_smiles),
        "novelty_among_unique": (
            len(novel_unique_smiles) / len(unique_smiles)
            if unique_smiles else 0.0
        ),
        "elapsed_seconds": round(time.time() - started, 2),
        "output_csv": str(output_path),
    }

    error_summary = {}
    for name in PROPERTIES:
        errors = [
            row[f"{name}_absolute_error"]
            for row in valid_rows
            if row[f"{name}_absolute_error"] is not None
        ]
        if errors:
            mean_deviation = sum(errors) / len(errors)
            error_summary[name] = {
                "n": len(errors),
                "mad": mean_deviation,
                "absolute_deviation_std": (
                    statistics.stdev(errors) if len(errors) > 1 else None
                ),
            }
    summary["property_error_summary"] = error_summary

    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("\n===== GENERATION SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print("CSV:", output_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
