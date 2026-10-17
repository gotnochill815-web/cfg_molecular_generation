"""
Model loading utilities.
Compatible with PyTorch Lightning + DDP + QLoRA.
"""

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def load_model(model_cfg):
    """
    Load tokenizer and QLoRA-ready model.
    """

    model_name = model_cfg["model_name"]

    ############################################################
    # Tokenizer
    ############################################################

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": model_cfg["special_tokens"]
        }
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    ############################################################
    # Quantization
    ############################################################

    quantization_config = None

    if model_cfg.get("load_in_4bit", False):

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    ############################################################
    # Load Model
    ############################################################

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None,   # IMPORTANT for Lightning DDP
    )

    ############################################################
    # Resize Embeddings
    ############################################################

    model.resize_token_embeddings(len(tokenizer))

    ############################################################
    # Disable KV cache
    ############################################################

    model.config.use_cache = False

    ############################################################
    # Gradient checkpointing
    ############################################################

    if model_cfg.get("gradient_checkpointing", False):

        model.gradient_checkpointing_enable()

    ############################################################
    # Prepare for QLoRA
    ############################################################

    if model_cfg.get("load_in_4bit", False):

        model = prepare_model_for_kbit_training(model)

    model.enable_input_require_grads()

    ############################################################
    # LoRA
    ############################################################

    lora_cfg = LoraConfig(
        r=model_cfg["lora"]["r"],
        lora_alpha=model_cfg["lora"]["alpha"],
        lora_dropout=model_cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(
        model,
        lora_cfg,
    )

    model.print_trainable_parameters()

    return model, tokenizer