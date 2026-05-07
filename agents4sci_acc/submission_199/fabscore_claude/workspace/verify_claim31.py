#!/usr/bin/env python3
"""Verify claim 31: lenient accuracy differences mostly negative (GPT-4o -10.0 pp, Gemini -8.1 pp)"""
import json
import os

LOGS_DIR = "/home/chenhui/fabscore/agent4sci_acc/submission_199/199_Multi_LLM_and_Multi_Prompt_Supplementary Material/Logs"

STRATEGIES = ["no_guide", "public_health_expert", "respiratory_doctor", "detailed_public_health", "detailed_respiratory"]
STRAT_LABELS = ["S1", "S2", "S3", "S4", "S5"]

def compute_metrics(records):
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for rec in records:
        label = rec.get("judgment_classification")
        try:
            label = int(label)
        except (TypeError, ValueError):
            continue
        if label in counts:
            counts[label] += 1
    N = sum(counts.values())
    strict = counts[1] / N * 100 if N > 0 else 0
    lenient = (counts[1] + counts[2]) / N * 100 if N > 0 else 0
    return {"strict": strict, "lenient": lenient, "N": N}

# Extract model names from filenames
all_files = [f for f in os.listdir(LOGS_DIR) if f.endswith(".json")]
models = set()
for f in all_files:
    for strat in STRATEGIES:
        suffix = f"_{strat}.json"
        if f.endswith(suffix):
            model = f[:-len(suffix)]
            models.add(model)
            break
models = sorted(models)
print(f"Found {len(models)} models: {models}\n")

# Compute per-model lenient accuracy per strategy
model_data = {}
for model in models:
    row = {}
    for strat in STRATEGIES:
        path = os.path.join(LOGS_DIR, f"{model}_{strat}.json")
        if not os.path.exists(path):
            row[strat] = None
            continue
        with open(path) as f:
            data = json.load(f)
        records = data if isinstance(data, list) else data.get("results", [])
        m = compute_metrics(records)
        row[strat] = m["lenient"]
    model_data[model] = row

# Print lenient accuracy table
print("Per-model lenient accuracy (%) by strategy:")
header = f"{'Model':<40} " + " ".join(f"{s:<8}" for s in STRAT_LABELS)
print(header)
print("-" * 80)
for model in models:
    vals = [f"{model_data[model][s]:.2f}" if model_data[model][s] is not None else "N/A" for s in STRATEGIES]
    print(f"{model:<40} " + " ".join(f"{v:<8}" for v in vals))

# S3-S2 comparison
print("\nS3 - S2 (respiratory_doctor - public_health_expert) differences:")
neg_count = 0
total_count = 0
for model in models:
    s2 = model_data[model].get("public_health_expert")
    s3 = model_data[model].get("respiratory_doctor")
    if s2 is not None and s3 is not None:
        total_count += 1
        diff = s3 - s2
        if diff < 0:
            neg_count += 1
        flag = " *" if diff < 0 else ""
        print(f"  {model:<40}: {s3:.2f}% - {s2:.2f}% = {diff:+.2f} pp{flag}")

print(f"\nModels with negative S3-S2 difference: {neg_count}/{total_count}")
if total_count > 0:
    print(f"'Mostly negative' check: {neg_count/total_count*100:.0f}%")

# Specific models
print("\nSpecific models mentioned in claim 31:")
for model_key, label in [("chatgpt-4o-latest", "GPT-4o"), ("gemini-2.5-pro", "Gemini")]:
    if model_key in model_data:
        s2 = model_data[model_key].get("public_health_expert")
        s3 = model_data[model_key].get("respiratory_doctor")
        if s2 and s3:
            diff = s3 - s2
            print(f"{label} ({model_key}): S3({s3:.2f}%) - S2({s2:.2f}%) = {diff:+.2f} pp (claim: -10.0 pp for GPT-4o, -8.1 pp for Gemini)")
        else:
            print(f"{label}: S2={s2}, S3={s3}")
    else:
        print(f"{label} ({model_key}): not found")
