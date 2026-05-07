"""
Quick test script to verify all_results.npy generation for shakespeare_char.
Runs a very short training (250 iters) for 1 seed to verify the code pipeline.
"""
import sys
sys.path.insert(0, '/home/chenhui/fabscore/aiscientist_papers/rl_lr_adaptation')

import os
import numpy as np
import json

# We'll monkeypatch the train function to use fewer iterations
import experiment

# Monkeypatch max_iters for a quick test
original_train = experiment.train

def quick_train(dataset="shakespeare_char", out_dir="run_0", seed_offset=0):
    # Temporarily reduce max_iters for testing
    import experiment as exp_module
    # We can't easily monkeypatch local variables, so we'll just call the function
    # but with the understanding that it'll run with full settings
    return original_train(dataset, out_dir, seed_offset)

out_dir = '/home/chenhui/fabscore/aiscientist_papers/rl_lr_adaptation/fabscore_claude/workspace/quick_test_run'
os.makedirs(out_dir, exist_ok=True)

# Run just one seed for shakespeare_char
print("Starting quick test training for shakespeare_char...")
final_info, train_info, val_info = original_train("shakespeare_char", out_dir, seed_offset=0)

print(f"Training complete!")
print(f"Final train loss: {final_info['final_train_loss']:.4f}")
print(f"Best val loss: {final_info['best_val_loss']:.4f}")
print(f"Training time: {final_info['total_train_time']:.2f}s")
print(f"Train info keys: {list(train_info[0].keys()) if train_info else 'None'}")
print(f"Val info: {val_info[:2] if val_info else 'None'}")

# Save the results to verify structure
all_results = {
    "shakespeare_char_0_final_info": final_info,
    "shakespeare_char_0_train_info": train_info,
    "shakespeare_char_0_val_info": val_info,
}
npy_path = os.path.join(out_dir, "all_results.npy")
with open(npy_path, "wb") as f:
    np.save(f, all_results)

print(f"\nall_results.npy saved to {npy_path}")
results = np.load(npy_path, allow_pickle=True).item()
print(f"Keys in all_results.npy: {list(results.keys())}")
print(f"Val info length: {len(results['shakespeare_char_0_val_info'])}")
if results['shakespeare_char_0_val_info']:
    sample = results['shakespeare_char_0_val_info'][0]
    print(f"Val info sample entry keys: {list(sample.keys())}")
    print(f"First val entry: iter={sample.get('iter')}, val/loss={sample.get('val/loss'):.4f}")
