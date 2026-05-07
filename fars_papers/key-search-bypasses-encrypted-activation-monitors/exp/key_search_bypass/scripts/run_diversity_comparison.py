# Compare key-induced embedding diversity between full (lambda2=0.5) and
# no-diversity (lambda2=0) encryptors. For 200 held-out prompts, sample 32 keys
# per prompt and compute pairwise L2 distances. Output violin plot + stats JSON.
import sys
import os
import json
import numpy as np
import torch
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_DIR))

from transformers import AutoModelForCausalLM, AutoTokenizer
from key_search_bypass.encryptor.model import KeyConditionedEncryptor
from key_search_bypass.data.alpaca import load_alpaca_subset

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
KEY_DIM = 128
N_PROMPTS = 200
N_KEYS = 32


def collect_pairwise_distances(model, encryptor, input_ids, attention_mask, n_prompts=200, n_keys=32):
    encryptor.eval()
    n = min(n_prompts, input_ids.shape[0])
    all_dists = []

    for i in range(n):
        ids = input_ids[i:i+1].cuda()
        mask = attention_mask[i:i+1].cuda()

        with torch.no_grad():
            clean_embeds = model.model.embed_tokens(ids)

        embeddings_list = []
        for _ in range(n_keys):
            key = KeyConditionedEncryptor.sample_key(1, KEY_DIM, device=ids.device, dtype=clean_embeds.dtype)
            with torch.no_grad():
                z = encryptor(clean_embeds, key)
            seq_len = mask.sum().item()
            z_mean = z[0, :int(seq_len)].mean(dim=0)
            embeddings_list.append(z_mean)

        z_stack = torch.stack(embeddings_list, dim=0).float()
        pairwise = torch.cdist(z_stack.unsqueeze(0), z_stack.unsqueeze(0), p=2).squeeze(0)
        triu_mask = torch.triu(torch.ones(n_keys, n_keys, device=pairwise.device), diagonal=1).bool()
        dists = pairwise[triu_mask]
        all_dists.append(dists.cpu())

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n} prompts")

    all_dists = torch.cat(all_dists)
    return all_dists.numpy()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f"Loading model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    hidden_dim = model.config.hidden_size

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    val_data = load_alpaca_subset(n=500, seed=9999, tokenizer=tokenizer)
    val_ids = torch.tensor(val_data["input_ids"], dtype=torch.long)
    val_mask = torch.tensor(val_data["attention_mask"], dtype=torch.long)

    variants = {}

    full_ckpt_dir = PROJ_DIR / "key_search_bypass" / "outputs" / "encryptor" / "seed_123_diverse"
    no_div_ckpt_dir = PROJ_DIR / "key_search_bypass" / "outputs" / "encryptor_no_div"

    best_no_div_kl = float("inf")
    best_no_div_path = None
    for seed in [42, 123, 456]:
        p = no_div_ckpt_dir / f"seed_{seed}" / "best_checkpoint.pt"
        if p.exists():
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
            kl = ckpt.get("eval_kl", float("inf"))
            if kl < best_no_div_kl:
                best_no_div_kl = kl
                best_no_div_path = p

    ckpt_configs = [
        ("Full (λ₂=0.5)", full_ckpt_dir / "best_checkpoint.pt"),
        ("No diversity (λ₂=0)", best_no_div_path),
    ]

    for label, ckpt_path in ckpt_configs:
        if ckpt_path is None or not ckpt_path.exists():
            print(f"Skipping {label}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\nLoading {label} from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        encryptor = KeyConditionedEncryptor(hidden_dim=hidden_dim, key_dim=KEY_DIM)
        encryptor.load_state_dict(ckpt["encryptor_state_dict"])
        encryptor = encryptor.to(dtype=torch.bfloat16, device="cuda")
        encryptor.eval()

        print(f"  Computing pairwise distances ({N_PROMPTS} prompts, {N_KEYS} keys)...")
        dists = collect_pairwise_distances(model, encryptor, val_ids, val_mask, N_PROMPTS, N_KEYS)
        mean_d = float(np.mean(dists))
        std_d = float(np.std(dists))
        print(f"  Mean pairwise L2: {mean_d:.4f} +/- {std_d:.4f}")

        variants[label] = {
            "distances": dists,
            "mean": mean_d,
            "std": std_d,
        }

        encryptor = encryptor.cpu()
        del encryptor
        torch.cuda.empty_cache()

    fig_dir = PROJ_DIR / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plot_data = []
    labels_list = []
    for label, info in variants.items():
        n_sample = min(50000, len(info["distances"]))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(info["distances"]), n_sample, replace=False)
        plot_data.append(info["distances"][idx])
        labels_list.append(label)

    parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(labels_list) + 1))
    ax.set_xticklabels(labels_list, fontsize=11)
    ax.set_ylabel("Pairwise L2 Distance", fontsize=12)
    ax.set_title("Key-Induced Embedding Diversity: Full vs No-Diversity Encryptor", fontsize=13)

    for i, (label, info) in enumerate(variants.items()):
        ax.text(i + 1, max(info["distances"][:50000]) * 0.95,
                f"μ={info['mean']:.3f}\nσ={info['std']:.3f}",
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    fig_path = fig_dir / "key_diversity_ablation.pdf"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved violin plot to {fig_path}")
    plt.close()

    stats = {}
    for label, info in variants.items():
        stats[label] = {"mean_pairwise_l2": info["mean"], "std_pairwise_l2": info["std"]}

    stats_path = PROJ_DIR / "key_search_bypass" / "results" / "diversity_comparison_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
