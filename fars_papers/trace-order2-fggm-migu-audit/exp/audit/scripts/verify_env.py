# Environment verification script: downloads Qwen2-1.5B, tests model loading/generation,
# verifies DeepSpeed, and checks all key package imports.
# Run on GPU via TrainService to validate the full stack.

import os
import sys
import json
import torch
from dotenv import load_dotenv

load_dotenv()

def verify_imports():
    print("=== Verifying package imports ===")
    packages = [
        "transformers", "datasets", "accelerate", "deepspeed",
        "peft", "safetensors", "tokenizers", "sentencepiece",
        "numpy", "scipy", "pandas", "matplotlib", "seaborn",
        "rouge_score", "nltk", "sacrebleu", "sklearn", "tqdm", "wandb",
        "flash_attn",
    ]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  [OK] {pkg}")
        except ImportError as e:
            print(f"  [FAIL] {pkg}: {e}")
    print()

def verify_cuda():
    print("=== Verifying CUDA ===")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    Memory: {mem:.1f} GB")
        print(f"  BF16 supported: {torch.cuda.is_bf16_supported()}")
    print()

def verify_deepspeed():
    print("=== Verifying DeepSpeed ===")
    import deepspeed
    print(f"  DeepSpeed version: {deepspeed.__version__}")
    ds_report = os.popen("ds_report 2>&1").read()
    for line in ds_report.split("\n"):
        if "compatible" in line.lower() or "version" in line.lower() or "status" in line.lower():
            print(f"  {line.strip()}")
    print()

def download_and_verify_model():
    print("=== Downloading and verifying Qwen2-1.5B ===")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2-1.5B"
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "base_models", "Qwen2-1.5B")
    os.makedirs(save_dir, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN", None)

    print(f"  Downloading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
    tokenizer.save_pretrained(save_dir)
    print(f"  Tokenizer saved to {save_dir}")

    print(f"  Downloading model from {model_name} in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        trust_remote_code=True,
    )
    model.save_pretrained(save_dir)
    print(f"  Model saved to {save_dir}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count / 1e9:.2f}B")
    print(f"  Model dtype: {next(model.parameters()).dtype}")

    if torch.cuda.is_available():
        print("  Moving model to GPU for generation test...")
        model = model.to("cuda:0")
        inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Generation test: '{generated}'")
    else:
        print("  No GPU available, skipping generation test")

    print()
    return save_dir

if __name__ == "__main__":
    verify_imports()
    verify_cuda()
    verify_deepspeed()
    model_dir = download_and_verify_model()
    print("=== All verifications complete ===")
    print(f"  Qwen2-1.5B saved to: {model_dir}")
