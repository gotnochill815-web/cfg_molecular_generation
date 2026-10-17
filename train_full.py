"""
Main training script using PyTorch Lightning.
"""

import argparse
import os
import shutil
from pprint import pprint

import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger

from src.utils.config import load_yaml
from src.training.load_model import load_model
from src.dataset.dataset import load_datasets
from src.training.collator import CFGDataCollator

from src.lightning.lightning_module import CFGLightningModule
from src.lightning.datamodule import CFGDataModule

import torch
from torch.optim import AdamW
import math


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset config yaml",
    )

    parser.add_argument(
        "--model",
        default="configs/model/olmo_7b.yaml",
    )

    parser.add_argument(
        "--training",
        default="configs/training/default.yaml",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Arguments")
    print(args)
    print("=" * 80)

    ###########################################################
    # Load Configs
    ###########################################################

    dataset_cfg = load_yaml(args.dataset)
    model_cfg = load_yaml(args.model)
    training_cfg = load_yaml(args.training)

    print("=" * 80)
    print("Dataset Config")
    pprint(dataset_cfg)

    print("=" * 80)
    print("Model Config")
    pprint(model_cfg)

    print("=" * 80)
    print("Training Config")
    pprint(training_cfg)

    ###########################################################
    # Seed
    ###########################################################

    L.seed_everything(training_cfg.get("seed", 42), workers=True)

    ###########################################################
    # Load model
    ###########################################################

    print("=" * 80)
    print("Loading Model")
    print("=" * 80)

    model, tokenizer = load_model(model_cfg)

    ###########################################################
    # Load datasets
    ###########################################################

    print("=" * 80)
    print("Loading Dataset")
    print("=" * 80)

    train_dataset, val_dataset, test_dataset = load_datasets(dataset_cfg)

    print(f"Train : {len(train_dataset)}")
    print(f"Valid : {len(val_dataset)}")

    ###########################################################
    # DataModule
    ###########################################################

    datamodule = CFGDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg.get("num_workers", 4),
        dropout_prob=training_cfg.get("dropout_prob", 0.2),
        max_length=dataset_cfg["max_length"],
        conditioning_mode=training_cfg["conditioning"]["mode"],
    )

    ###########################################################
    # Optimizer
    ###########################################################

    lr = float(training_cfg["learning_rate"])

    optimizer_partial = lambda params: AdamW(
        params,
        lr=lr,
        weight_decay=training_cfg.get("weight_decay", 0.01),
    )

    ###########################################################
    # Scheduler
    ###########################################################

    # The training DataLoader uses drop_last=True.
    steps_per_epoch = (
        len(train_dataset) // training_cfg["batch_size"]
    )

    accumulation = int(training_cfg["gradient_accumulation_steps"])
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accumulation)
    total_steps = optimizer_steps_per_epoch * int(training_cfg["epochs"])

    def scheduler_partial(optimizer):
        warmup_steps = int(training_cfg.get("warmup_steps", 0))
        min_lr = float(training_cfg["min_learning_rate"])

        if not (0.0 < min_lr <= lr):
            raise ValueError(
                f"Expected 0 < min_learning_rate <= learning_rate, got {min_lr}, {lr}"
            )

        min_ratio = min_lr / lr
        decay_steps = max(1, total_steps - warmup_steps)

        def lr_lambda(current_step):
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(warmup_steps)

            progress = (current_step - warmup_steps) / decay_steps
            progress = min(max(progress, 0.0), 1.0)

            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )

    ###########################################################
    # Lightning Module
    ###########################################################

    lightning_model = CFGLightningModule(
        model=model,
        optimizer_partial=optimizer_partial,
        scheduler_partials=[scheduler_partial],
    )

    ###########################################################
    # Logging
    ###########################################################

    logger = TensorBoardLogger(
        save_dir="logs",
        name="cfg",
    )

    ###########################################################
    # Callbacks
    ###########################################################

    callbacks = [

        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="best-{epoch:02d}",
        ),

        LearningRateMonitor(
            logging_interval="step"
        ),

        EarlyStopping(
            monitor="val/loss",
            patience=5,
            mode="min",
        ),
    ]

    ###########################################################
    # Trainer
    ###########################################################

    trainer = L.Trainer(

        accelerator="gpu",

        devices=training_cfg.get(
            "devices",
            1,
        ),

        strategy=training_cfg.get(
            "strategy",
            "auto",
        ),

        precision=training_cfg.get(
            "precision",
            "bf16-mixed",
        ),

        max_epochs=training_cfg["epochs"],

        accumulate_grad_batches=training_cfg.get(
            "gradient_accumulation_steps",
            1,
        ),

        gradient_clip_val=training_cfg.get(
            "max_grad_norm",
            1.0,
        ),

        log_every_n_steps=training_cfg.get(
            "logging_steps",
            10,
        ),

        callbacks=callbacks,

        logger=logger,
    )

    ###########################################################
    # Train
    ###########################################################

    print("=" * 80)
    print("Starting Lightning Training")
    print("=" * 80)

    trainer.fit(
        lightning_model,
        datamodule=datamodule,
    )

    ###########################################################
    # Save Everything
    ###########################################################

    save_dir = training_cfg["output_dir"]
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 80)
    print("Saving Everything...")
    print("=" * 80)

    # 1. Save LoRA adapter
    adapter_dir = os.path.join(save_dir, "adapter")
    model.save_pretrained(adapter_dir)

    # 2. Save tokenizer
    tokenizer_dir = os.path.join(save_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_dir)

    # 3. Save Lightning checkpoint
    trainer.save_checkpoint(
        os.path.join(save_dir, "last.ckpt")
    )

    # 4. Save model config
    config_dir = os.path.join(save_dir, "config")
    model.config.save_pretrained(config_dir)

    # 5. Save YAML configs
    shutil.copy(args.training, os.path.join(save_dir, "training.yaml"))
    shutil.copy(args.model, os.path.join(save_dir, "model.yaml"))
    shutil.copy(args.dataset, os.path.join(save_dir, "dataset.yaml"))

    print("=" * 80)
    print("Training Complete")
    print("=" * 80)
    print(f"Everything saved to: {save_dir}")


if __name__ == "__main__":
    main()
