# Classifier-Free Guidance for Molecular Generation

A research framework for molecular generation conditioned on molecular properties using classifier-free guidance (CFG).

The project is designed to evaluate different compatible causal language models and their fine-tuned adapters under a shared molecular-generation and evaluation pipeline.

## Key Features

- Classifier-free guidance using conditional and unconditional model predictions.
- Molecular property conditioning for QED, LogP, topological polar surface area (TPSA), and synthetic accessibility score (SAS).
- Configurable guidance scale, with a default of 2.0.
- LoRA-based fine-tuning and configurable model/tokenizer paths.
- SMILES validity, uniqueness, novelty, and property-error evaluation.
- Dataset-specific normalization statistics.
- Generation experiments using different models, adapters, datasets, and guidance scales.

## Method

CFG combines conditional and unconditional predictions:

`guided = unconditional + guidance_scale * (conditional - unconditional)`

The model receives a property-conditioned prompt and an unconditional prompt. The guidance scale controls the influence of the requested molecular properties during generation.

The default guidance scale is 2.0. It can be changed using `--guidance-scale`.

## Repository Structure

- `src/cfg/`: classifier-free guidance and decoding.
- `src/generation/`: molecular generation and guidance sweeps.
- `src/training/`: training components and LoRA utilities.
- `src/lightning/`: training modules and data handling.
- `src/dataset/`: dataset processing utilities.
- `src/evaluation/`: molecular evaluation metrics.
- `src/utils/`: configuration and property-scaling utilities.
- `configs/`: model, dataset, training, and generation configurations.
- `scripts/`: dataset normalization and experiment utilities.

Datasets, model weights, checkpoints, training logs, and generated output files are not included in this repository.

## Supported Molecular Properties

Single-property generation configurations are provided for:

- QED: `configs/generation/qed.json`
- LogP: `configs/generation/logp.json`
- TPSA: `configs/generation/tpsa.json`
- SAS: `configs/generation/sas.json`

Use the normalization statistics associated with the dataset and model used for the particular experiment. Do not reuse statistics from another dataset without verification.

## Configure a Model

Set paths to the base model, its matching trained CFG adapter, and tokenizer. These values are experiment-specific.

```bash
export MODEL_NAME="your-model-name"
export BASE_MODEL_PATH="/path/to/base-model"
export ADAPTER_DIR="/path/to/trained-cfg-adapter"
export TOKENIZER_DIR="/path/to/matching-tokenizer"
```

`MODEL_NAME` is a label for identifying experiments. The actual paths must point to the correct model, adapter, and tokenizer.

The base model must be compatible with the model-loading and CFG-decoding implementation. Other model architectures may require adaptations to the loader, tokenizer, training code, or special-token handling.

## Generate Molecules

Example using the QED configuration:

```bash
python -m src.generation.chemfm_cfg_generate \
  --generation-config configs/generation/qed.json \
  --base-model "$BASE_MODEL_PATH" \
  --adapter-dir "$ADAPTER_DIR" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --output-csv "outputs/generation/${MODEL_NAME}_qed_cfg.csv" \
  --guidance-scale 2.0
```

Replace `qed.json` with `logp.json`, `tpsa.json`, or `sas.json` to generate molecules for another property.

The matching training dataset must be available locally for novelty evaluation. The generation configuration and property-scaling statistics must match the selected trained model.

## Training and Experiments

Training requires a compatible base model, the appropriate dataset, and the corresponding dataset and training configurations.

For each experiment, record:

- Model and checkpoint identifier.
- Dataset and train/validation/test splits.
- Property normalization statistics.
- Training configuration and number of epochs.
- Guidance scale, temperature, top-p, and random seed.
- Validity, uniqueness, novelty, property error, and generation time.

Different models should be compared using consistent evaluation procedures and documented configurations.

## Reproducibility and Data

Large datasets, base-model weights, LoRA adapters, checkpoints, logs, and generated molecules should be stored separately from the source repository.

No training-completion status or benchmark performance is implied by this repository description. Report experimental results only after the corresponding runs have completed and been evaluated.
