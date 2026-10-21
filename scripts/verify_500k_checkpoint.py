import json
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path("checkpoints/lora_cfg_500000_chemfm_2epoch")
ADAPTER = ROOT / "adapter"
TOKENIZER = ROOT / "tokenizer"

STATS = Path("data/guacamol_500k/normalized/property_stats_500k.json")
STATIC_STATS = Path("configs/scaling/guacamol_500k.json")
TRAIN = Path("data/guacamol_500k/normalized/train_500k_refnorm.csv")

PROPS = ["qed", "logp", "tpsa", "sas"]
TOKENS = [
    "<QED>", "<LOGP>", "<TPSA>", "<SAS>",
    "<NULL>", "<pstart>", "<molstart>"
]

def report(label, passed, detail=""):
    status = "PASS" if passed else "CHECK"
    print(f"[{status}] {label}")
    if detail:
        print(f"        {detail}")

print("\n===== 500K CHECKPOINT AUDIT =====\n")

report("Adapter directory exists", ADAPTER.is_dir(), str(ADAPTER))
report("Tokenizer directory exists", TOKENIZER.is_dir(), str(TOKENIZER))

# 1. Tokenizer
if TOKENIZER.is_dir():
    tok = AutoTokenizer.from_pretrained(
        str(TOKENIZER),
        local_files_only=True
    )

    print("\n--- Special tokens ---")
    for token in TOKENS:
        ids = tok.encode(token, add_special_tokens=False)
        is_special = token in tok.all_special_tokens
        report(
            token,
            len(ids) == 1 and is_special,
            f"ids={ids}, special={is_special}"
        )

    print("Tokenizer size:", len(tok))

# 2. LoRA configuration
config_path = ADAPTER / "adapter_config.json"
if config_path.exists():
    config = json.loads(config_path.read_text())
    print("\n--- Adapter config ---")
    print("Base model:", config.get("base_model_name_or_path"))
    print("LoRA rank:", config.get("r"))
    print("Target modules:", config.get("target_modules"))
else:
    report("Adapter config exists", False)

# 3. Embedding weights
print("\n--- Saved adapter weights ---")
weight_files = sorted(ADAPTER.glob("*.safetensors"))
report("Safetensors files exist", bool(weight_files))

embedding_keys = []
lora_keys = []

for path in weight_files:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            lower = key.lower()
            if "lora_" in lower:
                lora_keys.append(key)
            if any(s in lower for s in [
                "embed_tokens", "wte", "lm_head",
                "word_embeddings"
            ]):
                embedding_keys.append(key)

report("LoRA weights found", bool(lora_keys), f"count={len(lora_keys)}")
report(
    "Embedding weights found in adapter",
    bool(embedding_keys),
    f"count={len(embedding_keys)}"
)

for key in embedding_keys[:10]:
    print("  ", key)

# 4. Normalization
print("\n--- Normalization statistics ---")
if STATS.exists() and STATIC_STATS.exists():
    train_stats = json.loads(STATS.read_text())["properties"]
    static_stats = json.loads(STATIC_STATS.read_text())["properties"]

    for prop in PROPS:
        a, b = train_stats[prop], static_stats[prop]
        same = (
            np.isclose(a["mean"], b["mean"], rtol=1e-9, atol=1e-9)
            and np.isclose(a["std"], b["std"], rtol=1e-9, atol=1e-9)
        )
        report(
            f"{prop} statistics match",
            same,
            f"train=({a['mean']:.6f}, {a['std']:.6f}), "
            f"config=({b['mean']:.6f}, {b['std']:.6f})"
        )
else:
    report(
        "Both 500k statistics files exist",
        False,
        f"train_stats={STATS.exists()}, static_stats={STATIC_STATS.exists()}"
    )

# 5. Training data distribution
print("\n--- Normalized training data ---")
if TRAIN.exists():
    df = pd.read_csv(TRAIN, usecols=PROPS)
    print("Training rows:", len(df))

    for prop in PROPS:
        mean = float(df[prop].mean())
        std = float(df[prop].std(ddof=0))
        passed = abs(mean) < 0.01 and abs(std - 1.0) < 0.01
        report(
            f"{prop} approximately standard-normal",
            passed,
            f"mean={mean:.6f}, std={std:.6f}"
        )
else:
    report("Normalized training CSV exists", False, str(TRAIN))

print("\n===== AUDIT FINISHED =====")
