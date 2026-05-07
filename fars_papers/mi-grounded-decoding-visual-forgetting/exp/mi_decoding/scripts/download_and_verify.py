# Download and verify models and datasets for MI decoding experiments.
# Run on GPU to verify model loading with Qwen2VLForConditionalGeneration.
import os
import gc
import sys
import torch
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

def download_and_verify_model(model_id, device="cuda"):
    print(f"\n{'='*60}")
    print(f"Downloading and verifying: {model_id}")
    print(f"{'='*60}")
    from huggingface_hub import snapshot_download
    path = snapshot_download(model_id)
    print(f"Downloaded to: {path}")

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print(f"Loading model on {device}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device,
    )
    print(f"Model config:\n  hidden_size={model.config.hidden_size}")
    print(f"  num_layers={model.config.num_hidden_layers}")
    print(f"  vocab_size={model.config.vocab_size}")
    print(f"  model_type={model.config.model_type}")

    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Processor loaded: {type(processor).__name__}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Model {model_id} verified and unloaded.")
    return path

def download_and_verify_datasets():
    print(f"\n{'='*60}")
    print("Downloading and verifying datasets")
    print(f"{'='*60}")

    from datasets import load_dataset
    print("\nLoading MMStar...")
    ds = load_dataset("Lin-Chen/MMStar")
    for split in ds:
        print(f"  MMStar split '{split}': {len(ds[split])} samples")
        print(f"  Columns: {ds[split].column_names}")
        sample = ds[split][0]
        print(f"  Sample keys: {list(sample.keys())}")
    del ds

    print("\nCloning HallusionBench...")
    hb_dir = os.path.join(os.path.dirname(__file__), "../../HallusionBench")
    if not os.path.exists(hb_dir):
        os.system(f"git clone https://github.com/tianyi-lab/HallusionBench.git {hb_dir}")
    else:
        print(f"HallusionBench already cloned at {hb_dir}")

    hb_json = os.path.join(hb_dir, "HallusionBench.json")
    if os.path.exists(hb_json):
        import json
        with open(hb_json) as f:
            hb_data = json.load(f)
        print(f"  HallusionBench: {len(hb_data)} samples loaded from JSON")
        if hb_data:
            print(f"  Sample keys: {list(hb_data[0].keys())}")
    else:
        print(f"  WARNING: {hb_json} not found, checking directory contents...")
        for item in os.listdir(hb_dir)[:20]:
            print(f"    {item}")

    print("\nDataset verification complete.")

if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    download_and_verify_datasets()

    if torch.cuda.is_available():
        p1 = download_and_verify_model("Qwen/Qwen2.5-VL-7B-Instruct", device="cuda")
        p2 = download_and_verify_model("UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B", device="cuda")
    else:
        print("No GPU available, skipping model loading verification")
        from huggingface_hub import snapshot_download
        p1 = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")
        p2 = snapshot_download("UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B")
        print(f"Qwen2.5-VL-7B-Instruct downloaded to: {p1}")
        print(f"VLAA-Thinker downloaded to: {p2}")

    print("\n" + "="*60)
    print("ALL DOWNLOADS AND VERIFICATIONS COMPLETE")
    print("="*60)
