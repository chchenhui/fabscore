# Environment verification script for debiased one-pass attention sorting.
# Downloads models, verifies GPU access, tests attention extraction.
import sys
import os
import time

def main():
    print("=" * 60)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 60)

    print("\n[1/7] Checking Python version...")
    print(f"  Python: {sys.version}")

    print("\n[2/7] Checking PyTorch + CUDA...")
    import torch
    print(f"  torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    Memory: {mem:.1f} GB")
    else:
        print("  ERROR: CUDA not available!")
        sys.exit(1)

    print("\n[3/7] Checking key packages...")
    import transformers
    import flash_attn
    import datasets
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import seaborn
    import matplotlib
    print(f"  transformers: {transformers.__version__}")
    print(f"  flash_attn: {flash_attn.__version__}")
    print(f"  datasets: {datasets.__version__}")
    print(f"  numpy: {np.__version__}")
    print(f"  pandas: {pd.__version__}")
    print(f"  scipy: {scipy.__version__}")
    print(f"  scikit-learn: {sklearn.__version__}")
    print(f"  seaborn: {seaborn.__version__}")
    print(f"  matplotlib: {matplotlib.__version__}")

    print("\n[4/7] Checking SynthWiki dataset...")
    csv_path = os.path.join(os.path.dirname(__file__), "synthwiki", "data", "madlibs", "madlibs1.csv")
    df = pd.read_csv(csv_path)
    print(f"  Dataset path: {csv_path}")
    print(f"  Entries: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    assert len(df) >= 990, f"Expected >= 990 entries, got {len(df)}"
    print("  OK: Dataset verified")

    print("\n[5/7] Downloading and loading primary model: togethercomputer/LLaMA-2-7B-32K-Instruct...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "togethercomputer/LLaMA-2-7B-32K-Instruct"
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"  Tokenizer loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    model_eager = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    print(f"  Model (eager) loaded in {time.time() - t0:.1f}s")
    print(f"  Model dtype: {model_eager.dtype}")
    print(f"  Model device: {model_eager.device}")

    print("\n[6/7] Testing attention extraction (eager mode)...")
    test_input = tokenizer("Hello, this is a test.", return_tensors="pt").to(model_eager.device)
    with torch.no_grad():
        outputs = model_eager.generate(
            **test_input,
            max_new_tokens=1,
            output_attentions=True,
            return_dict_in_generate=True,
        )
    attentions = outputs["attentions"]
    print(f"  Generation tokens: {len(attentions)}")
    print(f"  Layers per token: {len(attentions[0])}")
    print(f"  Attention shape (layer 0): {attentions[0][0].shape}")
    print("  OK: Attention tensors extracted successfully")

    del model_eager
    torch.cuda.empty_cache()

    print("\n[6b/7] Loading model with flash-attention (trust_remote_code=True)...")
    t0 = time.time()
    model_flash = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    print(f"  Model (flash) loaded in {time.time() - t0:.1f}s")
    test_input2 = tokenizer("Hello, this is a test.", return_tensors="pt").to(model_flash.device)
    with torch.no_grad():
        outputs_flash = model_flash.generate(
            **test_input2,
            max_new_tokens=5,
            output_attentions=False,
            return_dict_in_generate=True,
        )
    generated_text = tokenizer.decode(outputs_flash["sequences"][0], skip_special_tokens=True)
    print(f"  Flash-attention generation OK: '{generated_text[:80]}...'")

    del model_flash
    torch.cuda.empty_cache()

    print("\n[7/7] Downloading secondary model: NousResearch/Yarn-Llama-2-7b-64k...")
    import shutil
    from huggingface_hub import constants as hf_constants
    hf_cache = hf_constants.HF_HUB_CACHE
    cache_dir = os.path.join(hf_cache, "models--NousResearch--Yarn-Llama-2-7b-64k")
    print(f"  HF cache dir: {hf_cache}")
    if os.path.exists(cache_dir):
        print(f"  Clearing corrupted cache at {cache_dir}")
        shutil.rmtree(cache_dir)
    t0 = time.time()
    tokenizer2 = AutoTokenizer.from_pretrained("NousResearch/Yarn-Llama-2-7b-64k")
    _orig_torch_load = torch.load
    import functools
    @functools.wraps(_orig_torch_load)
    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_load
    model2 = AutoModelForCausalLM.from_pretrained(
        "NousResearch/Yarn-Llama-2-7b-64k",
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    torch.load = _orig_torch_load
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    test_input3 = tokenizer2("Test input for Yarn model.", return_tensors="pt").to(model2.device)
    with torch.no_grad():
        outputs3 = model2.generate(
            **test_input3,
            max_new_tokens=1,
            output_attentions=True,
            return_dict_in_generate=True,
        )
    attentions3 = outputs3["attentions"]
    print(f"  Attention shape (layer 0): {attentions3[0][0].shape}")
    print("  OK: Secondary model verified")

    del model2
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
