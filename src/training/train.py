"""
Trainer builder.
"""

import os

from transformers import (
    Trainer,
    TrainingArguments,
)


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    collator,
    training_cfg,
):

    # ------------------------------------------------------------------
    # Create checkpoint directory
    # ------------------------------------------------------------------

    output_dir = os.path.abspath(training_cfg["output_dir"])

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("TrainingArguments output_dir:")
    print(output_dir)
    print("=" * 80)

    # ------------------------------------------------------------------
    # Training Arguments
    # ------------------------------------------------------------------

    args = TrainingArguments(

        output_dir=output_dir,

        # Training
        per_device_train_batch_size=training_cfg["batch_size"],
        per_device_eval_batch_size=training_cfg["batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],

        num_train_epochs=training_cfg["epochs"],

        learning_rate=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
        warmup_steps=training_cfg["warmup_steps"],

        # Logging
        logging_strategy="steps",
        logging_steps=training_cfg["logging_steps"],

        # Saving
        save_strategy="steps",
        save_steps=training_cfg["save_steps"],
        save_total_limit=training_cfg["save_total_limit"],
        save_only_model=True,

        # Evaluation
        eval_strategy="no",

        # Precision
        fp16=training_cfg["fp16"],
        bf16=training_cfg["bf16"],

        # Misc
        remove_unused_columns=False,
        report_to="none",
    )

    print("=" * 80)
    print("Trainer Output Directory")
    print(args.output_dir)
    print("=" * 80)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    return trainer