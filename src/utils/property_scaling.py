"""Property normalization shared by training-data scaling and generation.

Statistics are always loaded from an explicit statistics file so that a run can
never silently normalize targets with another dataset's mean/std. The files
live in ``configs/scaling/`` and are produced by
``scripts/normalize_guacamol_500k.py`` alongside the normalized CSVs.
"""

import json
import math
from pathlib import Path

PROPERTIES = ("qed", "logp", "tpsa", "sas")

# Relative tolerance used when comparing two statistics files.
STATS_TOLERANCE = 1e-9


def load_property_stats(stats_file):
    """Load {property: (mean, std)} from a statistics JSON file."""
    if stats_file is None:
        raise ValueError(
            "A normalization statistics file is required. Pass the file that "
            "matches the adapter being used, e.g. "
            "--stats-file configs/scaling/guacamol_500k.json"
        )

    path = Path(stats_file).expanduser()
    with path.open() as handle:
        payload = json.load(handle)

    rows = payload.get("properties", payload)
    result = {}
    for name in PROPERTIES:
        item = rows.get(name)
        if not isinstance(item, dict) or "mean" not in item or "std" not in item:
            raise ValueError(f"Statistics file has no valid mean/std for {name}")
        mean, std = float(item["mean"]), float(item["std"])
        if std <= 0:
            raise ValueError(f"Standard deviation for {name} must be positive")
        result[name] = (mean, std)
    return result


def find_dataset_stats(training_csv):
    """Return the statistics file saved next to a normalized training CSV."""
    directory = Path(training_csv).expanduser().parent
    candidates = sorted(directory.glob("property_stats*.json"))
    return candidates[0] if candidates else None


def verify_stats_match_dataset(property_stats, training_csv):
    """Fail when the inference statistics differ from the dataset's own file.

    The normalized CSVs are written together with the statistics that produced
    them, so a mismatch means the prompts would be built on a different scale
    than the one the model was trained on.
    """
    dataset_stats_path = find_dataset_stats(training_csv)
    if dataset_stats_path is None:
        print(
            "WARNING: no property_stats*.json next to "
            f"{training_csv}; could not cross-check normalization statistics."
        )
        return None

    dataset_stats = load_property_stats(dataset_stats_path)

    for name in PROPERTIES:
        mean, std = property_stats[name]
        dataset_mean, dataset_std = dataset_stats[name]
        if not (
            math.isclose(mean, dataset_mean, rel_tol=STATS_TOLERANCE, abs_tol=0.0)
            and math.isclose(std, dataset_std, rel_tol=STATS_TOLERANCE, abs_tol=0.0)
        ):
            raise ValueError(
                f"Normalization statistics for {name} do not match the dataset "
                f"used for training: inference mean/std = {mean}/{std}, "
                f"{dataset_stats_path} mean/std = {dataset_mean}/{dataset_std}"
            )

    print("Normalization statistics match", dataset_stats_path)
    return dataset_stats_path


def normalize_targets(targets, property_stats):
    """Convert raw property targets to the scale used during training."""
    result = {}
    for name, value in targets.items():
        if name not in property_stats:
            raise ValueError(f"Unknown property: {name}")
        if value is None:
            result[name] = None
            continue
        mean, std = property_stats[name]
        result[name] = (float(value) - mean) / std
    return result


def denormalize_properties(values, property_stats):
    """Convert normalized property values back to original units."""
    result = {}
    for name, value in values.items():
        if name not in property_stats:
            raise ValueError(f"Unknown property: {name}")
        if value is None:
            result[name] = None
            continue
        mean, std = property_stats[name]
        result[name] = float(value) * std + mean
    return result
