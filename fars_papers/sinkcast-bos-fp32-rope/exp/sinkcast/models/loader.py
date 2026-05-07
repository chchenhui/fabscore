# Unified model/tokenizer loading for Llama-3.1-8B and Mistral-7B-v0.3.
# Supports BF16 and FP32 dtypes, HF Hub authentication via HF_TOKEN env var.

import os
from typing import Tuple

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


SUPPORTED_MODELS = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b-v0.3": "mistralai/Mistral-7B-v0.3",
}


def resolve_model_name(model_name: str) -> str:
    return SUPPORTED_MODELS.get(model_name.lower(), model_name)


def load_model_and_tokenizer(
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    attn_implementation: str = "flash_attention_2",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    load_dotenv()
    token = os.environ.get("HF_TOKEN", None)

    model_path = resolve_model_name(model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        token=token,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        token=token,
        attn_implementation=attn_implementation,
    )
    model.eval()

    return model, tokenizer
