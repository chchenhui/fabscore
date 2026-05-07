# Shared training utilities: seeding, config loading, checkpoint saving, helpers.
# Used by sequential_trainer.py and all methods.

import os
import random
import yaml
import json
import torch
import numpy as np
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            output[k] = v.to(device)
        else:
            output[k] = v
    return output


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg, flush=True)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def save_hf_format(model, tokenizer, output_dir, sub_folder=""):
    save_dir = os.path.join(output_dir, sub_folder) if sub_folder else output_dir
    os.makedirs(save_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)


def load_model_and_tokenizer(model_path: str, bf16: bool = True, gradient_checkpointing: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        trust_remote_code=True,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model, tokenizer


def get_optimizer_grouped_parameters(model, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    return [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
