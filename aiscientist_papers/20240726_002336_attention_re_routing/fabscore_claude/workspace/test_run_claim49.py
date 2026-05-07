"""
Minimal test to verify claim 49: run shakespeare_char training from run_1
to generate all_results.npy and verify Figure 1 can be reproduced.
Only runs shakespeare_char dataset with 1 seed to keep it fast.
"""
import os
import sys
import json
import numpy as np

# Add repo root to path
REPO_ROOT = "/home/chenhui/fabscore/aiscientist_papers/20240726_002336_attention_re_routing"
WORKSPACE = "/home/chenhui/fabscore/aiscientist_papers/20240726_002336_attention_re_routing/fabscore_claude/workspace"
OUT_DIR = os.path.join(WORKSPACE, "test_run_1")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, REPO_ROOT)

# Import the train function from run_1
import importlib.util
spec = importlib.util.spec_from_file_location("run_1", os.path.join(REPO_ROOT, "run_1.py"))
run1_mod = importlib.util.load_from_spec(spec)
spec.loader.exec_module(run1_mod)

# Run just shakespeare_char with 1 seed
print("Starting shakespeare_char training (seed 0)...")
final_info, train_info, val_info = run1_mod.train("shakespeare_char", OUT_DIR, seed_offset=0)

# Save results
all_results = {}
all_results["shakespeare_char_0_final_info"] = final_info
all_results["shakespeare_char_0_train_info"] = train_info
all_results["shakespeare_char_0_val_info"] = val_info

out_npy = os.path.join(OUT_DIR, "all_results_shakespeare_only.npy")
with open(out_npy, "wb") as f:
    np.save(f, all_results)

print(f"\nSaved all_results to {out_npy}")
print(f"train_info keys: {list(train_info[0].keys()) if isinstance(train_info, list) and train_info else 'N/A'}")
print(f"val_info length: {len(val_info) if val_info else 0}")
print(f"final_info: {final_info}")

# Check the structure
loaded = np.load(out_npy, allow_pickle=True).item()
print(f"\nLoaded all_results keys: {list(loaded.keys())}")
key = "shakespeare_char_0_val_info"
if key in loaded:
    vinfo = loaded[key]
    print(f"val_info type: {type(vinfo)}, len: {len(vinfo) if vinfo else 0}")
    if vinfo:
        print(f"First val entry: {vinfo[0]}")
        print(f"Last val entry: {vinfo[-1]}")
