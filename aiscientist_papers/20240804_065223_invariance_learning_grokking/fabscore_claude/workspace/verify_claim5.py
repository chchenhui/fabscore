"""Minimal verification script for Claim 5 (Figure 2d: Permutation dynamics).
Runs permutation with seed 0 to get training loss, val accuracy, and invariance score over time.
"""

import sys
import os
import json

repo_root = "/home/chenhui/fabscore/aiscientist_papers/20240804_065223_invariance_learning_grokking"
sys.path.insert(0, repo_root)
os.chdir(repo_root)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load run_4.py definitions (stop before main block)
exec_globals = {}
with open(os.path.join(repo_root, "run_4.py"), "r") as f:
    source = f.read()

lines = source.split('\n')
main_start = None
for i, line in enumerate(lines):
    if line.strip().startswith('parser = argparse') or line.strip() == 'if __name__ == "__main__":':
        main_start = i
        break

func_source = '\n'.join(lines[:main_start]) if main_start is not None else source
exec(func_source, exec_globals)
run = exec_globals['run']

out_dir = "/home/chenhui/fabscore/aiscientist_papers/20240804_065223_invariance_learning_grokking/fabscore_claude/workspace"
dataset = "permutation"
seed_offset = 0

print(f"Running {dataset} with seed offset {seed_offset}")
print("This verifies the learning dynamics for Figure 2(d): Permutations")

final_info, train_info, val_info, invariance_scores = run(out_dir, dataset, seed_offset)

print(f"\n=== Results for {dataset}, seed {seed_offset} ===")
print(f"Final train loss: {final_info['final_train_loss']:.6f}")
print(f"Final val loss: {final_info['final_val_loss']:.6f}")
print(f"Final train acc: {final_info['final_train_acc']:.4f}")
print(f"Final val acc: {final_info['final_val_acc']:.4f}")
print(f"Grokking point: {final_info['grokking_point']}")
print(f"Invariance point: {final_info['invariance_point']}")

print(f"\nFirst 5 training steps:")
for info in train_info[:5]:
    print(f"  step={info['step']}: train_loss={info['train_loss']:.4f}, train_acc={info['train_accuracy']:.4f}")

print(f"\nFirst 5 validation steps:")
for info in val_info[:5]:
    print(f"  step={info['step']}: val_loss={info['val_loss']:.4f}, val_acc={info['val_accuracy']:.4f}")

print(f"\nFirst 5 invariance scores:")
for info in invariance_scores[:5]:
    print(f"  step={info['step']}: invariance_score={info['score']:.4f}")

print(f"\nLast 5 invariance scores:")
for info in invariance_scores[-5:]:
    print(f"  step={info['step']}: invariance_score={info['score']:.4f}")

# Save results
results = {
    "final_info": final_info,
    "train_info_sample": train_info[:20],
    "val_info_sample": val_info[:20],
    "invariance_scores_sample": invariance_scores[:20],
    "grokking_step_val_info": [v for v in val_info if v.get('val_accuracy', 0) >= 0.99][:5],
}

with open(os.path.join(out_dir, "claim5_verification_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to claim5_verification_results.json")

# Plot figure to verify Figure 2(d) content
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

steps_train = [info['step'] for info in train_info]
train_losses = [info['train_loss'] for info in train_info]

steps_val = [info['step'] for info in val_info]
val_accs = [info['val_accuracy'] for info in val_info]

steps_inv = [info['step'] for info in invariance_scores]
inv_scores = [info['score'] for info in invariance_scores]

axes[0].plot(steps_train, train_losses, 'b-', label='Train Loss')
axes[0].set_ylabel('Training Loss')
axes[0].set_title('Training Loss (permutation, seed 0)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(steps_val, val_accs, 'g-', label='Val Accuracy')
axes[1].set_ylabel('Validation Accuracy')
axes[1].set_title('Validation Accuracy (permutation, seed 0)')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(steps_inv, inv_scores, 'r-', label='Invariance Score')
axes[2].set_ylabel('Invariance Score')
axes[2].set_xlabel('Training Steps')
axes[2].set_title('Invariance Score (permutation, seed 0)')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "claim5_figure2d_verification.png"))
print(f"Figure saved to claim5_figure2d_verification.png")
