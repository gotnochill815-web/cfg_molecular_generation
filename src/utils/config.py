from pathlib import Path
import yaml

# Find the project root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path):
    """
    Load a YAML configuration file relative to the project root.
    """

    config_path = PROJECT_ROOT / path

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg