"""Generate 36 LLaMA-Factory YAML configs for proxy model (Qwen2.5-1.5B) LoRA SFT
at tiny learning rate (5e-6) across 12 datasets x 3 seeds.
Single-GPU training, 500 steps, periodic eval every 100 steps, best-checkpoint selection.
Only difference from gen_proxy_std_configs.py: learning_rate=5e-6, output/run paths use proxy_tiny."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TLR_ROOT = PROJECT_ROOT / "tlr_proxy_sft"
CONFIG_DIR = TLR_ROOT / "configs" / "proxy_tiny_lr"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "AM-Thinking-v1-Distilled-math",
    "DeepMath-309K",
    "Maths-College",
    "OpenR1-Math",
    "QwQ-LongCoT-130K-math",
    "R1-Distill-SFT-math",
    "hkust-nlp__dart-math-hard",
    "mathplus",
    "numinamath-cot",
    "numinamath1_5",
    "openmathinstruct-2",
    "Magpie-Reasoning-V2-250K-CoT-QwQ-math",
]

SEEDS = [42, 123, 456]

TEMPLATE = """\
### model
model_name_or_path: Qwen/Qwen2.5-1.5B
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

### dataset
dataset: {dataset_name}
dataset_dir: {dataset_dir}
template: default
cutoff_len: 4096
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: {output_dir}
logging_steps: 10
save_steps: 100
save_total_limit: 5
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb
run_name: proxy_tiny_{dataset_name}_seed{seed}

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
max_steps: 500
learning_rate: 5.0e-6
lr_scheduler_type: cosine
warmup_ratio: 0.1
weight_decay: 0.0
bf16: true
use_liger_kernel: true
packing: false
seed: {seed}

### eval (periodic evaluation + best checkpoint)
val_size: 0.02
per_device_eval_batch_size: 4
eval_strategy: steps
eval_steps: 100
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
"""


def main():
    dataset_dir = str(TLR_ROOT / "data")
    count = 0

    for ds in DATASETS:
        for seed in SEEDS:
            output_dir = str(TLR_ROOT / "outputs" / "proxy_tiny" / ds / f"seed_{seed}")
            config = TEMPLATE.format(
                dataset_name=ds,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                seed=seed,
            )
            fname = f"{ds}_seed{seed}.yaml"
            config_path = CONFIG_DIR / fname
            with open(config_path, "w") as f:
                f.write(config)
            count += 1

    print(f"Generated {count} config files in {CONFIG_DIR}")


if __name__ == "__main__":
    main()
