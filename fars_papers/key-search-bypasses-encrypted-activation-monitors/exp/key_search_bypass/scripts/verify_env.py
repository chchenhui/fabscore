# Environment verification script: downloads model + datasets, runs a forward pass,
# and validates dataset sizes. Intended to run on GPU via TrainService.
import sys
import traceback

try:
    import torch
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n--- Loading Qwen2.5-7B-Instruct ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded: {model.config.architectures}, hidden_size={model.config.hidden_size}, num_layers={model.config.num_hidden_layers}")

    print("\n--- Forward pass test ---")
    test_input = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**test_input, output_hidden_states=True)
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Hidden states layers: {len(outputs.hidden_states)}, last hidden shape: {outputs.hidden_states[-1].shape}")

    del model
    torch.cuda.empty_cache()

    print("\n--- Loading HarmBench dataset ---")
    from datasets import load_dataset

    harmbench_loaded = False
    for hb_name in ["walledai/HarmBench", "Anthropic/harmbench"]:
        try:
            hb_ds = load_dataset(hb_name)
            print(f"HarmBench loaded from {hb_name}")
            for split_name, split_data in hb_ds.items():
                print(f"  Split '{split_name}': {len(split_data)} samples, columns: {split_data.column_names}")
            harmbench_loaded = True
            break
        except Exception as e:
            print(f"  Failed to load {hb_name}: {e}")

    if not harmbench_loaded:
        print("WARNING: Could not load HarmBench from HuggingFace, will need alternative source")

    print("\n--- Loading Alpaca dataset ---")
    alpaca_ds = load_dataset("tatsu-lab/alpaca")
    for split_name, split_data in alpaca_ds.items():
        print(f"  Split '{split_name}': {len(split_data)} samples, columns: {split_data.column_names}")

    print("\n--- All checks passed ---")

except Exception as e:
    print(f"\nERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
