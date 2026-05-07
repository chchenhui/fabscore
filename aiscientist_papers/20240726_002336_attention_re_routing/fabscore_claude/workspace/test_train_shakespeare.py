"""
Minimal wrapper to run just shakespeare_char training from run_1.py
and verify the all_results.npy output format for claim 49 verification.
"""
import os
import sys
import json
import numpy as np

REPO_ROOT = "/home/chenhui/fabscore/aiscientist_papers/20240726_002336_attention_re_routing"
WORKSPACE = "/home/chenhui/fabscore/aiscientist_papers/20240726_002336_attention_re_routing/fabscore_claude/workspace"
OUT_DIR = os.path.join(WORKSPACE, "test_run_1_shakespeare")
os.makedirs(OUT_DIR, exist_ok=True)

# We need to run from the repo root for relative data path to work
os.chdir(REPO_ROOT)

# Use exec to load the module without triggering argparse
with open(os.path.join(REPO_ROOT, "run_1.py"), "r") as f:
    code = f.read()

# Remove the argparse lines and the if __name__ == "__main__" block
# by splitting at the argparse section
split_marker = "\nparser = argparse.ArgumentParser"
if split_marker in code:
    code_without_main = code[:code.index(split_marker)]
else:
    # try another approach
    code_without_main = code

# Create a namespace and exec
ns = {"__name__": "run_module"}
exec(compile(code_without_main, "run_1.py", "exec"), ns)

train_fn = ns["train"]

# Run only shakespeare_char, seed 0
print("=" * 60)
print("Running shakespeare_char training (seed_offset=0)...")
print("=" * 60)

final_info, train_info, val_info = train_fn("shakespeare_char", OUT_DIR, seed_offset=0)

print("\n" + "=" * 60)
print("Training completed!")
print(f"final_info: {final_info}")
print(f"train_info type: {type(train_info)}, length: {len(train_info) if train_info else 'None'}")
print(f"val_info type: {type(val_info)}, length: {len(val_info) if val_info else 'None'}")

if train_info:
    print(f"First train entry: {train_info[0]}")
    print(f"Last train entry: {train_info[-1]}")
if val_info:
    print(f"First val entry: {val_info[0]}")
    print(f"Last val entry: {val_info[-1]}")

# Build all_results dict as done in the original script
all_results = {}
all_results["shakespeare_char_0_final_info"] = final_info
all_results["shakespeare_char_0_train_info"] = train_info
all_results["shakespeare_char_0_val_info"] = val_info

out_npy = os.path.join(OUT_DIR, "all_results.npy")
with open(out_npy, "wb") as f:
    np.save(f, all_results)

print(f"\nSaved all_results.npy to {out_npy}")

# Verify the file loads correctly
loaded = np.load(out_npy, allow_pickle=True).item()
print(f"Loaded keys: {list(loaded.keys())}")

val_key = "shakespeare_char_0_val_info"
if val_key in loaded and loaded[val_key]:
    vinfo = loaded[val_key]
    iters = [info["iter"] for info in vinfo]
    val_losses = [info["val/loss"] for info in vinfo]
    print(f"\nNumber of validation checkpoints: {len(iters)}")
    print(f"Iteration range: {iters[0]} to {iters[-1]}")
    print(f"Val loss range: {min(val_losses):.4f} to {max(val_losses):.4f}")
    print(f"\nFinal val loss: {val_losses[-1]:.4f}")
    print(f"Final train loss (from final_info): {final_info.get('final_train_loss', 'N/A')}")

print("\nDONE - all_results.npy format verified for claim 49")
