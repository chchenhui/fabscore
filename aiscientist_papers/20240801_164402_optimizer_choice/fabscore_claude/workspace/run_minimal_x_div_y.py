"""
Minimal verification script: run AdamW on x_div_y for 1 seed
to verify training accuracy curves are produceable.
Run from: /home/chenhui/fabscore/aiscientist_papers/20240801_164402_optimizer_choice/
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, '/home/chenhui/fabscore/aiscientist_papers/20240801_164402_optimizer_choice')

# Import the run function from experiment.py
from experiment import run

out_dir = '/home/chenhui/fabscore/aiscientist_papers/20240801_164402_optimizer_choice/fabscore_claude/workspace/minimal_run'
os.makedirs(out_dir, exist_ok=True)

print("Running AdamW on x_div_y, seed 0 (7500 steps) ...")
final_info, train_log_info, val_log_info = run(out_dir, 'x_div_y', 0, 'AdamW')

print("=== Final Info ===")
print(json.dumps(final_info, indent=2))

# Print training accuracy at key checkpoints
print("\n=== Training Accuracy Time Series (every 500 steps) ===")
for entry in train_log_info:
    if entry['step'] % 500 == 0 or entry['step'] <= 100:
        print(f"  step={entry['step']:5d}: train_acc={entry['train_accuracy']:.4f}")

print("\n=== Validation Accuracy Time Series (every 500 steps) ===")
for entry in val_log_info:
    if entry['step'] % 500 == 0 or entry['step'] <= 100:
        print(f"  step={entry['step']:5d}: val_acc={entry['val_accuracy']:.4f}")

# Save the time-series data
result = {
    'final_info': final_info,
    'train_log_info': train_log_info,
    'val_log_info': val_log_info,
}
np.save(os.path.join(out_dir, 'minimal_x_div_y_adamw_seed0.npy'), result)
print(f"\nSaved results to {out_dir}/minimal_x_div_y_adamw_seed0.npy")
print("Done.")
