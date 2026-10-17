"""Convert raw GuacaMol property targets to training-normalized values."""

PROPERTY_STATS = {
    "qed":  (0.5527237460854895, 0.21418527743796437),
    "logp": (3.4789038467500024, 1.8179559370311646),
    "sas":  (2.8893985600731518, 0.8088543337407695),
    "tpsa": (80.67809025, 38.739946383034074),
}


def normalize_targets(**targets):
    """Normalize raw requested property values using training statistics."""
    result = {}

    for name, value in targets.items():
        if name not in PROPERTY_STATS:
            raise ValueError(f"Unknown property: {name}")

        if value is None:
            result[name] = None
            continue

        mean, std = PROPERTY_STATS[name]
        result[name] = (float(value) - mean) / std

    return result


def denormalize_properties(**values):
    """Convert normalized property values back to original units."""
    result = {}

    for name, value in values.items():
        if name not in PROPERTY_STATS:
            raise ValueError(f"Unknown property: {name}")

        if value is None:
            result[name] = None
            continue

        mean, std = PROPERTY_STATS[name]
        result[name] = float(value) * std + mean

    return result
