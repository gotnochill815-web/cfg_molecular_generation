# Classifier-Free Guidance (CFG) for Molecular Generation

A research framework for molecular generation conditioned on molecular properties using classifier-free guidance (CFG).

The project is designed to evaluate compatible causal language models and fine-tuned adapters through a shared molecular-generation and evaluation pipeline. The included OLMo-7B GuacaMol-derived experiment is one setup, not a claim that every causal language model can be used without adapting the model loader, tokenizer, special tokens, or decoding implementation.

## Key Features

- Classifier-free guidance combining conditional and unconditional model predictions.
- Property-conditioned generation for QED, LogP, topological polar surface area (TPSA), and synthetic accessibility score (SAS).
- LoRA-based fine-tuning with configurable model and tokenizer paths.
- Dataset-specific z-score normalization for continuous conditioning properties.
- Evaluation of SMILES validity, uniqueness, novelty, property error, and generation time.
- Configurable generation targets, guidance scale, sampling settings, and experiment output paths.

## Method

CFG combines conditional and unconditional model predictions:

```text
guided = unconditional + guidance_scale * (conditional - unconditional)
```

The decoder evaluates the conditional and unconditional prompts and combines their predictions during generation. The guidance scale controls how strongly the conditional prediction influences the result. A scale of `1.0` corresponds to the conditional prediction under this formula; larger values extrapolate further in the conditional direction and can trade validity or diversity for property control.

Guidance scales are experiment-dependent. The repository's preliminary single-property starting points are LogP `1.5`, TPSA `2.0`, QED `2.5`, and SAS `2.5`. These are not universal optima. For multi-property generation, pass `--guidance-scale` explicitly rather than relying on a default.

## Repository Structure

- `src/cfg/`: classifier-free guidance and decoding.
- `src/generation/`: molecular generation runners and guidance sweeps.
- `src/training/`: training components and LoRA utilities.
- `src/lightning/`: training modules and data handling.
- `src/dataset/`: dataset-processing modules.
- `src/evaluation/`: molecular evaluation metrics.
- `src/utils/`: utilities, including legacy property-scaling constants.
- `configs/generation/`: property-target generation configurations.
- `configs/scaling/`: dataset-specific inference normalization statistics.
- `configs/dataset/`: dataset paths and property-column mappings used for training.
- `configs/training/`: training settings.
- `scripts/`: data-normalization and experiment scripts.
- `jobs/`: example Slurm job scripts for cluster runs.

Datasets, model weights, adapters, checkpoints, training logs, and generated outputs are stored separately and are not included in the source repository.

## Supported Molecular Properties

Single-property generation configurations are provided for:

- QED: `configs/generation/qed.json`
- LogP: `configs/generation/logp.json`
- TPSA: `configs/generation/tpsa.json`
- SAS: `configs/generation/sas.json`

Multi-property configurations can be added using the same JSON format. For example, the local TPSA+SAS experiment uses `configs/generation/tpsa_sas.json`.

### Preliminary guidance scales and target values

| Property | Suggested CFG scale | Example target values |
|---|---:|---|
| LogP | 1.5 | 2.0, 4.0, 6.0 |
| TPSA | 2.0 | 40.0, 80.0, 120.0 |
| QED | 2.5 | 0.3, 0.5, 0.7 |
| SAS | 2.5 | 2.0, 3.0, 4.0 |

These choices came from preliminary small sweeps on one GuacaMol-derived 500K setup. Re-evaluate them when the model, dataset, or sampling parameters change. Target values are expressed in the original property units; the generation runner normalizes these values before constructing the conditioned prompt.

## Property Normalization

**The training CSV contains normalized property columns. Generation target values should be supplied in raw property units, and the generation runner must use the same mean and standard deviation that were used to normalize the training data.** Do not normalize the target values yourself and then pass them to the runner, because that would normalize them twice.

### Formula

The project uses z-score normalization:

```text
normalized_value = (raw_value - training_mean) / training_std
```

For the current `guacamol_500k` setup, the means and population standard deviations (`ddof=0`) are calculated from the raw training split only. The same training-derived values are then applied to the training, validation, and test splits. The validation or test split must not be used to fit these statistics.

### Current `guacamol_500k` statistics

The current training split contains 400,000 rows. The checked-in inference statistics file for this experiment is `configs/scaling/guacamol_500k.json`.

| Property | Training mean | Training standard deviation |
|---|---:|---:|
| QED | 0.5347159994971383 | 0.21524785224362858 |
| LogP | 3.439410031474989 | 2.031079091235181 |
| TPSA | 82.65614447500113 | 44.695205557151624 |
| SAS | 2.9279175352717473 | 0.84168601358627 |

The equivalent statistics generated alongside the normalized CSV are recorded at `data/guacamol_500k/normalized/property_stats_500k.json`. In the audited cluster setup, its values match `configs/scaling/guacamol_500k.json`.

### Relevant files

- Raw training data: `data/guacamol_500k/raw/train_500k.csv`
- Raw validation data: `data/guacamol_500k/raw/validation_500k.csv`
- Raw test data: `data/guacamol_500k/raw/test_500k.csv`
- Normalized training data: `data/guacamol_500k/normalized/train_500k_refnorm.csv`
- Normalized validation data: `data/guacamol_500k/normalized/val_500k_refnorm.csv`
- Normalized test data: `data/guacamol_500k/normalized/test_500k_refnorm.csv`
- Normalization script: `scripts/normalize_guacamol_500k.py`
- Saved data statistics: `data/guacamol_500k/normalized/property_stats_500k.json`
- Inference statistics: `configs/scaling/guacamol_500k.json`
- Training dataset configuration: `configs/dataset/guacamol_500k_normalized.yaml`

### Recreate the normalized CSVs

The normalization script expects the three raw split files listed above. It computes statistics from `raw/train_500k.csv`, then applies those same training-only statistics to every split. It refuses to overwrite an existing normalized output directory.

```bash
python scripts/normalize_guacamol_500k.py
```

Only run this when `data/guacamol_500k/normalized/` does not already exist, or after safely moving/backing up the existing output directory. Inspect the raw files and outputs before replacing any experiment data.

### Important inference rule

Always pass the matching statistics file explicitly when generating molecules:

```bash
--stats-file configs/scaling/guacamol_500k.json
```

The file `src/utils/property_scaling.py` contains a separate set of legacy fallback constants that exactly match `configs/scaling/guacamol_50k.json`, not the 500K statistics. Some generic generation scripts call this utility directly and default to the 50K adapter; that is consistent with those scripts' defaults. The current `src/generation/chemfm_cfg_generate.py` runner uses those fallback constants only when no `--stats-file` is supplied. When using the `guacamol_500k` adapter, always pass the matching 500K statistics file explicitly and verify that any custom script does the same.

The generation JSON contains **raw target values** such as TPSA `80` and SAS `3`. The runner converts these values using the specified statistics before conditioning the model. Evaluation property values and absolute property errors are reported in the original/raw property units.

### Normalization verification and dataset provenance

For the current cluster copy, the normalization formula was independently checked against all 400,000 training rows. The raw and normalized rows had matching SMILES order, and the maximum absolute difference between the stored normalized values and the recomputed z-scores was approximately `1e-15` for each property. A 1,000-molecule descriptor audit also found the stored QED, LogP, TPSA, and SAS values reproducible with the installed RDKit implementation within the CSV's rounding precision.

These checks support that the current z-score transformation is internally consistent. They do not, by themselves, establish how the 500K-derived files were originally selected or explain why their raw descriptor statistics differ from other datasets or implementations.

#### Official GuacaMol partition audit

An exact-SMILES membership audit was run against the official GuacaMol v1 train, validation, and test files. The official files contain 1,273,104 train, 79,568 validation, and 238,706 test molecules; their exact-string intersections are zero.

| Local split | Rows | Exact matches in official combined collection |
|---|---:|---:|
| Local train | 400,000 | 399,679 (99.92%) |
| Local validation | 50,000 | 49,963 (99.93%) |
| Local test | 50,000 | 49,955 (99.91%) |

The local training split's official-partition membership was:

| Official partition | Exact matches in local train |
|---|---:|
| Official train | 320,119 |
| Official validation | 19,885 |
| Official test | 59,675 |
| **Official validation + test combined** | **79,560 (19.89% of local train)** |

The three local splits have zero pairwise exact-SMILES overlap. However, local train contains 19,885 molecules that exactly match official validation entries and 59,675 that exactly match official test entries. This strongly indicates that the local files are sampled from the combined GuacaMol collection rather than preserving the official partition boundaries, although the membership audit alone cannot prove the exact data-preparation procedure. The unmatched rows (321 train, 37 validation, 45 test) were not resolved by this exact-string check, and equivalent structures written as different SMILES have not been exhaustively checked by canonicalization.

**Benchmarking implication:** describe these files as a custom 500K GuacaMol-derived dataset/split. Do not describe its local test set as the untouched official GuacaMol test partition or claim a strict official split-preserving benchmark reproduction. In particular, official test molecules were included in local training. For a strict official-split evaluation, construct a separate dataset that samples training molecules only from official train and retains official validation/test membership.

#### Retrospective inference-normalization audit

A retrospective audit of 34 available generation summary/CSV pairs associated with the 500K adapter compared recorded raw targets against their stored normalized target columns. The audited LogP, QED, SAS, and TPSA target values matched the guacamol_500k z-score formula to floating-point precision (maximum observed discrepancies were approximately 5.6e-17 for LogP, 8.4e-17 for QED, 1.4e-17 for SAS, and 1.2e-16 for TPSA). For these audited target fields, the values matched the 500K statistics, not the 50K fallback constants. This is evidence for the audited outputs, not a guarantee about uninspected future runs.

Record the data source, selection procedure, descriptor implementation/version, split sizes, and normalization-statistics file when making benchmark comparisons. Do not replace the current model's normalization statistics with another dataset's statistics without verifying compatibility with the data used during training.

## Configure a Model

Set the base model, trained adapter, tokenizer, normalization statistics, and training SMILES reference. The paths below match the local cluster experiment and may need to be adjusted for another environment.

```bash
export MODEL_NAME="olmo-7b-cfg-guacamol-500k"
export BASE_MODEL_PATH="/home/intern1/demo/demo_model"
export ADAPTER_DIR="checkpoints/lora_cfg_500000_chemfm_2epoch/adapter"
export TOKENIZER_DIR="checkpoints/lora_cfg_500000_chemfm_2epoch/tokenizer"
export STATS_FILE="configs/scaling/guacamol_500k.json"
export TRAINING_SMILES_CSV="data/guacamol_500k/normalized/train_500k_refnorm.csv"
```

`MODEL_NAME` is an experiment label only. The selected adapter must be compatible with the chosen base model and tokenizer. The training SMILES CSV is used as the reference for novelty calculations; use the intended training split and document it when comparing experiments.

## Generate Molecules

### Single-property example: QED

```bash
python -m src.generation.chemfm_cfg_generate \
  --generation-config configs/generation/qed.json \
  --base-model "$BASE_MODEL_PATH" \
  --adapter-dir "$ADAPTER_DIR" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --stats-file "$STATS_FILE" \
  --training-smiles-csv "$TRAINING_SMILES_CSV" \
  --output-csv "outputs/generation/${MODEL_NAME}_qed_cfg.csv" \
  --guidance-scale 2.5
```

Replace `qed.json` with `logp.json`, `tpsa.json`, or `sas.json` for another single property. Explicitly set the guidance scale when reproducing a specific experiment.

### Multi-property example: TPSA + SAS

The local production target grid has nine combinations: TPSA `{40, 80, 120}` crossed with SAS `{2, 3, 4}`. A configuration for 999 requested samples uses 111 samples per combination:

```json
{
  "properties": {
    "TPSA+SAS": {
      "sample_points": [
        [40, 2], [40, 3], [40, 4],
        [80, 2], [80, 3], [80, 4],
        [120, 2], [120, 3], [120, 4]
      ],
      "num_samples": 111,
      "temperature": 1.0
    }
  }
}
```

Save this as a separate file, for example `configs/generation/tpsa_sas_production_999.json`; do not overwrite an existing pilot or sweep configuration. A matching generation command is:

```bash
mkdir -p outputs/generation/tpsa_sas_production_999

python -u -m src.generation.chemfm_cfg_generate \
  --generation-config configs/generation/tpsa_sas_production_999.json \
  --base-model "$BASE_MODEL_PATH" \
  --adapter-dir "$ADAPTER_DIR" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --stats-file "$STATS_FILE" \
  --training-smiles-csv "$TRAINING_SMILES_CSV" \
  --output-csv "outputs/generation/tpsa_sas_production_999/cfg_scale_3.0.csv" \
  --guidance-scale 3.0 \
  --top-p 1.0 \
  --source-max-len 512 \
  --target-max-len 512 \
  --samples-per-target 111 \
  --seed 42 \
  --verbose
```

The command requests `9 x 111 = 999` generation attempts. The number of valid molecules may be lower if some generated SMILES are invalid. The command intentionally omits `--max-new-tokens`, allowing the runner's default total-length budget to apply based on the source and target length settings. Check `python -m src.generation.chemfm_cfg_generate --help` for the options supported by the local checkout.

### Slurm cluster runs

Use a dedicated Slurm script for long experiments, request a GPU partition/resource available on the target cluster, and write logs and results to a new experiment-specific path. Example resource directives for a cluster with an A100 GPU partition are:

```bash
#SBATCH --job-name=tpsas999
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
```

Partition names, GPU types, memory limits, and time limits vary between clusters. Confirm them with the local scheduler before submitting. Avoid overwriting previous output CSVs.

## Evaluation

The generation runner evaluates molecules using RDKit-based descriptors and records the requested conditions and measured property values. Common metrics include:

- **Validity:** fraction of attempts that yield parseable, valid molecules.
- **Uniqueness:** fraction of valid molecules that are unique under the runner's canonicalization rules.
- **Novelty:** fraction of generated canonical SMILES not found in the supplied training SMILES reference.
- **Property MAE:** mean absolute difference between each requested raw target and the measured raw property value for valid generated molecules.
- **Generation time:** elapsed time for a run or sweep.

Report the denominator for each metric, whether uniqueness is calculated within a target group or across the whole run, which training reference is used for novelty, and how invalid outputs are handled. For a fair comparison between models, use the same target combinations, number of attempts, RDKit property definitions, canonicalization and deduplication rules, novelty reference, guidance scale, temperature, top-p, and token budget wherever possible.

Generation over a grid of requested properties is a conditional-generation evaluation, not automatically a held-out test-set evaluation. Do not describe it as test-set performance unless a separate held-out evaluation was performed.

## Training and Reproducibility

The local 500K-derived adapter's training configuration is `configs/training/default_500k_chemfm_2epoch.yaml`, and the normalized dataset configuration is `configs/dataset/guacamol_500k_normalized.yaml`. Verify the config files, model identifiers, and actual dataset paths used for a run rather than relying on experiment names alone.

For each experiment, record:

- Base-model ID/revision and adapter/checkpoint path.
- Dataset source, selection/sampling procedure, split sizes, and SMILES preprocessing.
- Descriptor implementation and RDKit version.
- Property normalization statistics and the file used at inference.
- Training configuration, random seed, number of epochs, and adapter settings.
- Guidance scale, target grid, temperature, top-p, token budget, and generation seed.
- Number of attempted and valid molecules, validity, uniqueness, novelty, property errors, and elapsed time.

Large datasets, model weights, adapters, checkpoints, logs, and generated molecules should remain outside the source repository when appropriate. Commit lightweight configs and scripts needed to reproduce experiments, but do not commit credentials or large private artifacts.

## Scope and Limitations

This is a research framework. Guidance scale recommendations are preliminary, and results depend on the model, adapter, data construction, property definitions, normalization, and decoding settings. The OLMo CFG setup is not an exact reproduction of the ChemFM architecture or benchmark protocol. Report only results from experiments that actually completed and were evaluated; repository configuration alone does not imply training or generation completed successfully.
