# SFT: Standard sequential fine-tuning with cross-entropy loss, no gradient masking.
# This is the simplest baseline - no hooks, no replay, just standard training.
# The sequential_trainer.py handles the training loop; SFT simply does nothing extra.

def get_hooks(model_engine, task_idx, task_name, config):
    return None
