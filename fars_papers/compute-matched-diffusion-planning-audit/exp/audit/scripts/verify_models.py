"""Verify that Dream-v0-Base-7B and Qwen2.5-7B are loadable and functional on GPU.
Loads each model in bfloat16, checks key methods, tests tokenizer, then frees GPU memory.
"""

import gc
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."

def verify_dream():
    print("=" * 60)
    print("Verifying Dream-org/Dream-v0-Base-7B ...")
    model_id = "Dream-org/Dream-v0-Base-7B"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    encoded = tokenizer.encode(SAMPLE_TEXT)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"  Tokenizer encode: {len(encoded)} tokens")
    print(f"  Tokenizer decode: '{decoded}'")
    assert len(encoded) > 0, "Tokenizer produced empty encoding"
    assert SAMPLE_TEXT in decoded, f"Round-trip mismatch: '{decoded}'"

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"  Model loaded: {type(model).__name__}")
    print(f"  Model device: {next(model.parameters()).device}")
    assert hasattr(model, "diffusion_generate"), (
        "Model missing diffusion_generate method"
    )
    assert callable(model.diffusion_generate), (
        "diffusion_generate is not callable"
    )
    print("  diffusion_generate: callable OK")

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Parameters: {param_count:.2f}B")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("  GPU memory released")
    print("  Dream verification PASSED")
    print()


def verify_qwen():
    print("=" * 60)
    print("Verifying Qwen/Qwen2.5-7B ...")
    model_id = "Qwen/Qwen2.5-7B"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    encoded = tokenizer.encode(SAMPLE_TEXT)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"  Tokenizer encode: {len(encoded)} tokens")
    print(f"  Tokenizer decode: '{decoded}'")
    assert len(encoded) > 0, "Tokenizer produced empty encoding"
    assert SAMPLE_TEXT in decoded, f"Round-trip mismatch: '{decoded}'"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"  Model loaded: {type(model).__name__}")
    print(f"  Model device: {next(model.parameters()).device}")
    assert hasattr(model, "generate"), "Model missing generate method"
    assert callable(model.generate), "generate is not callable"
    print("  generate: callable OK")

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Parameters: {param_count:.2f}B")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("  GPU memory released")
    print("  Qwen verification PASSED")
    print()


def main():
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("ERROR: No CUDA device found. Exiting.")
        sys.exit(1)
    print()

    verify_dream()
    verify_qwen()

    print("=" * 60)
    print("ALL VERIFICATIONS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
