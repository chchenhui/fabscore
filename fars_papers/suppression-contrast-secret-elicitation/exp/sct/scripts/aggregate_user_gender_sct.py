# Aggregate SCT results for User Gender/Direct: combine token recovery + auditor
# results with bootstrap 95% CIs. Output format matches taboo_direct_sct.json.

import json
import os
import numpy as np

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
os.chdir(PROJECT_ROOT)

tr_path = "sct/results/user_gender_direct_sct_token_recovery.json"
aud_path = "sct/results/user_gender_direct_sct_auditor.json"
out_path = "sct/results/user_gender_direct_sct.json"

with open(tr_path) as f:
    tr_data = json.load(f)
with open(aud_path) as f:
    aud_data = json.load(f)


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95):
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, alpha * 100))
    hi = float(np.percentile(boot_means, (1 - alpha) * 100))
    return [round(lo, 4), round(hi, 4)]


def bootstrap_ci_binary(correct_list, n_bootstrap=10000, ci=0.95):
    arr = np.array(correct_list, dtype=float)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, alpha * 100))
    hi = float(np.percentile(boot_means, (1 - alpha) * 100))
    return [round(lo, 4), round(hi, 4)]


per_model = {}
genders = ["female", "male"]

for i, gender in enumerate(genders):
    tr_info = tr_data["per_model"][i]
    aud_info = aud_data["per_model"][i]

    guesses = aud_data["all_guesses"][gender]
    correct_arr = [1.0 if g["correct"] else 0.0 for g in guesses]
    aud_ci = bootstrap_ci_binary(correct_arr)

    scored_file = f"sct/outputs/user_gender_{gender}_sct_scored.json"
    with open(scored_file) as f:
        scored_data = json.load(f)

    secret_ids = set(tr_info["secret_token_ids"])
    tr5_hits = []
    tr20_hits = []
    for item in scored_data["results"]:
        if "error" in item:
            continue
        ranked = item.get("ranked_tokens", [])
        top5_ids = {t["token_id"] for t in ranked[:5]}
        top20_ids = {t["token_id"] for t in ranked[:20]}
        tr5_hits.append(1.0 if top5_ids & secret_ids else 0.0)
        tr20_hits.append(1.0 if top20_ids & secret_ids else 0.0)

    tr5_ci = bootstrap_ci_binary(tr5_hits)
    tr20_ci = bootstrap_ci_binary(tr20_hits)

    per_model[gender] = {
        "secret_word": gender,
        "secret_token_ids": tr_info["secret_token_ids"],
        "secret_token_strs": tr_info["secret_token_strs"],
        "num_examples": tr_info["num_examples"],
        "token_recovery_at_5": float(tr_info["token_recovery"]["5"]),
        "token_recovery_at_5_ci": tr5_ci,
        "token_recovery_at_20": float(tr_info["token_recovery"]["20"]),
        "token_recovery_at_20_ci": tr20_ci,
        "auditor_accuracy": aud_info["accuracy"],
        "auditor_accuracy_ci": aud_ci,
        "auditor_correct": aud_info["correct"],
        "auditor_total": aud_info["total"],
    }

tr5_vals = [per_model[g]["token_recovery_at_5"] for g in genders]
tr20_vals = [per_model[g]["token_recovery_at_20"] for g in genders]
aud_vals = [per_model[g]["auditor_accuracy"] for g in genders]

output = {
    "method": "sct",
    "description": "Suppression-Contrast Tokens on User Gender/Direct: score_SCT(v) = (1/T) * sum_t [log p_L(v|t) - log p_N(v|t)]. Ranks tokens suppressed between mid and final layers.",
    "settings": {
        "mid_layer": 32,
        "final_layer": 41,
        "alpha": 0.1,
        "top_k_stored": 200,
        "num_prompts": 100,
        "responses_per_prompt": 10,
        "total_examples_per_model": 1000,
        "auditor_model": "google/gemma-3-4b-it",
        "auditor_num_tokens": 5,
        "base_model": "google/gemma-2-9b-it",
    },
    "per_model": per_model,
    "summary": {
        "token_recovery_at_5_mean": float(np.mean(tr5_vals)),
        "token_recovery_at_5_std": float(np.std(tr5_vals)),
        "token_recovery_at_20_mean": float(np.mean(tr20_vals)),
        "token_recovery_at_20_std": float(np.std(tr20_vals)),
        "auditor_accuracy_mean": float(np.mean(aud_vals)),
        "auditor_accuracy_std": float(np.std(aud_vals)),
    },
}

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved to {out_path}")
print(f"\nSummary:")
print(f"  Token-recovery@5:  {output['summary']['token_recovery_at_5_mean']:.4f} +/- {output['summary']['token_recovery_at_5_std']:.4f}")
print(f"  Token-recovery@20: {output['summary']['token_recovery_at_20_mean']:.4f} +/- {output['summary']['token_recovery_at_20_std']:.4f}")
print(f"  Auditor accuracy:  {output['summary']['auditor_accuracy_mean']:.4f} +/- {output['summary']['auditor_accuracy_std']:.4f}")
print(f"\nPer-model:")
for g in genders:
    m = per_model[g]
    print(f"  {g}: TR@5={m['token_recovery_at_5']:.3f} CI={m['token_recovery_at_5_ci']}, "
          f"TR@20={m['token_recovery_at_20']:.3f} CI={m['token_recovery_at_20_ci']}, "
          f"Aud={m['auditor_accuracy']:.3f} CI={m['auditor_accuracy_ci']}")
