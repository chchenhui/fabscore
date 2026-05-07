"""Run just x_plus_y, seed 0, from experiment.py (run_0 baseline) and save training log."""
import sys
import os
import json

sys.path.insert(0, '/home/chenhui/fabscore/aiscientist_papers/data_augmentation_grokking')
WORKSPACE = '/home/chenhui/fabscore/aiscientist_papers/data_augmentation_grokking/fabscore_claude/workspace'

def load_run_fn(script_path):
    with open(script_path, 'r') as f:
        source = f.read()
    main_idx = source.find("if __name__ == '__main__':")
    if main_idx == -1:
        main_idx = source.find('if __name__ == "__main__":')
    source_trimmed = source[:main_idx] if main_idx != -1 else source
    ns = {'__name__': 'run_module', '__file__': script_path}
    exec(compile(source_trimmed, script_path, 'exec'), ns)
    return ns['run']

REPO = '/home/chenhui/fabscore/aiscientist_papers/data_augmentation_grokking'

out_dir = os.path.join(WORKSPACE, 'fig2_run0')
os.makedirs(out_dir, exist_ok=True)

run0_fn = load_run_fn(os.path.join(REPO, 'experiment.py'))
print("Running run_0 (baseline) x_plus_y seed 0...")
final0, train0, val0 = run0_fn(out_dir, 'x_plus_y', 0)
print(f"DONE. Final info: {final0}")

# Save step-by-step val accuracy
steps = [info['step'] for info in val0]
val_accs = [info['val_accuracy'] for info in val0]
print(f"Number of steps recorded: {len(steps)}")
print(f"First 5 steps: {steps[:5]}")
print(f"First 5 val_accs: {val_accs[:5]}")
print(f"Last 5 steps: {steps[-5:]}")
print(f"Last 5 val_accs: {val_accs[-5:]}")

# Save training log
log_data = {
    'steps': steps,
    'val_accs': val_accs,
    'final_info': final0,
}
with open(os.path.join(WORKSPACE, 'run0_xplusy_log.json'), 'w') as f:
    json.dump(log_data, f)
print("Saved run0_xplusy_log.json")
