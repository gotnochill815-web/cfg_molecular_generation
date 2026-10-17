from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch


def load_model(adapter_path):

    BASE_MODEL = "harindhar10/OLMo-7B-fsdp-Pubchem-2.5M-1epochs-eos"

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(
        model,
        adapter_path,
    )

    model.eval()

    return model, tokenizer