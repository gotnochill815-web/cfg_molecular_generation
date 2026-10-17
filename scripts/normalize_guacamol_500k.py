import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/guacamol_500k/raw"
OUT = ROOT / "data/guacamol_500k/normalized"

PROPERTIES = ["qed", "logp", "tpsa", "sas"]

SPLITS = {
    "train_500k.csv": "train_500k_refnorm.csv",
    "validation_500k.csv": "val_500k_refnorm.csv",
    "test_500k.csv": "test_500k_refnorm.csv",
}


def inspect_csv(path):
    """Validate schema and calculate online training statistics."""
    state = {
        p: {"n": 0, "mean": 0.0, "m2": 0.0}
        for p in PROPERTIES
    }
    row_count = 0
    fieldnames = None

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError(f"No CSV header found: {path}")

        missing = set(["smiles"] + PROPERTIES) - set(fieldnames)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            row_count += 1

            if not row.get("smiles", "").strip():
                raise ValueError(f"{path.name}: empty SMILES at line {line_number}")

            for prop in PROPERTIES:
                try:
                    x = float(row[prop])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{path.name}: invalid {prop} at line {line_number}: "
                        f"{row.get(prop)!r}"
                    )

                if not math.isfinite(x):
                    raise ValueError(
                        f"{path.name}: non-finite {prop} at line {line_number}"
                    )

                s = state[prop]
                s["n"] += 1
                delta = x - s["mean"]
                s["mean"] += delta / s["n"]
                delta2 = x - s["mean"]
                s["m2"] += delta * delta2

    if row_count == 0:
        raise ValueError(f"No data rows found in {path}")

    return fieldnames, row_count, state


def main():
    if not RAW.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {RAW}")

    # Refuse to overwrite previous normalized datasets.
    if OUT.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUT}\n"
            "Inspect it before rerunning. Nothing was overwritten."
        )

    for filename in SPLITS:
        if not (RAW / filename).is_file():
            raise FileNotFoundError(f"Missing input: {RAW / filename}")

    # Calculate statistics using the training split only.
    train_path = RAW / "train_500k.csv"
    train_fields, train_rows, state = inspect_csv(train_path)

    stats = {}
    for prop, s in state.items():
        mean = s["mean"]
        # Population standard deviation, consistent with z-score scaling.
        std = math.sqrt(s["m2"] / s["n"])

        if not math.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid standard deviation for {prop}: {std}")

        stats[prop] = {
            "mean": mean,
            "std": std,
            "n": s["n"],
        }

    # Validate split schemas before creating any output files.
    split_info = {}
    for filename in SPLITS:
        fields, nrows, _ = inspect_csv(RAW / filename)

        if fields != train_fields:
            raise ValueError(
                f"Column names/order differ between {filename} and train_500k.csv"
            )

        split_info[filename] = nrows

    # Normalize every split with training-only statistics.
    OUT.mkdir(parents=True, exist_ok=False)

    for input_name, output_name in SPLITS.items():
        source = RAW / input_name
        destination = OUT / output_name

        with source.open("r", newline="") as fin, \
             destination.open("x", newline="") as fout:

            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=train_fields)
            writer.writeheader()

            written = 0
            for row in reader:
                for prop in PROPERTIES:
                    x = float(row[prop])
                    row[prop] = format(
                        (x - stats[prop]["mean"]) / stats[prop]["std"],
                        ".17g",
                    )

                writer.writerow(row)
                written += 1

        if written != split_info[input_name]:
            raise RuntimeError(
                f"Row count mismatch for {input_name}: {written}"
            )

        print(f"Saved {destination} ({written:,} rows)")

    stats_output = {
        "dataset": "guacamol_500k",
        "method": "z_score",
        "formula": "(x - mean_train) / std_train",
        "fit_split": "train_500k.csv",
        "standard_deviation": "population (ddof=0)",
        "n_train_rows": train_rows,
        "properties": stats,
    }

    stats_path = OUT / "property_stats_500k.json"
    with stats_path.open("x") as f:
        json.dump(stats_output, f, indent=2)

    print(f"\nSaved training statistics: {stats_path}")
    print("\n===== TRAINING STATISTICS =====")
    for prop, values in stats.items():
        print(
            f"{prop:>5}: mean={values['mean']:.10f}, "
            f"std={values['std']:.10f}"
        )

    print("\n===== NORMALIZATION COMPLETE =====")
    print(f"Output directory: {OUT}")
    print("The raw input files were not modified.")


if __name__ == "__main__":
    main()
