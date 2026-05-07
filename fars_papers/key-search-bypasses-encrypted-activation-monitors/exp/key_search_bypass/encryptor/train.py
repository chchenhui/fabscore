# Encryptor training loop: trains the key-conditioned MLP encryptor on Alpaca
# prompts using utility-gated curriculum scheduling. The frozen Qwen2.5-7B-Instruct
# model is used for KL-based utility loss. Supports multi-seed runs, periodic eval,
# best-checkpoint saving, and WandB logging.
# max_steps counts OPTIMIZER steps (not micro-steps).

import os
import sys
import json
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import KeyConditionedEncryptor
from .losses import utility_loss, privacy_loss, diversity_loss, _forward_from_embeds
from .schedule import UtilityGatedScheduler

PROJ_DIR = Path(__file__).resolve().parents[2]
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--key_dim", type=int, default=128)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=0.5)
    parser.add_argument("--tau_low", type=float, default=0.005)
    parser.add_argument("--tau_high", type=float, default=0.05)
    parser.add_argument("--curriculum_warmup", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_data(tokenizer, train_size, val_size, max_length):
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    texts = [ex["instruction"] for ex in ds]

    rng = np.random.RandomState(seed=999)
    all_indices = rng.permutation(len(texts))
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]

    train_texts = [texts[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]

    train_enc = tokenizer(train_texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    val_enc = tokenizer(val_texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    return train_enc, val_enc


@torch.no_grad()
def evaluate(encryptor, model, val_input_ids, val_attention_mask, eval_samples, key_dim, batch_size=4):
    encryptor.eval()
    n = min(eval_samples, val_input_ids.shape[0])
    total_kl = 0.0
    total_tokens = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ids = val_input_ids[start:end].cuda()
        mask = val_attention_mask[start:end].cuda()

        clean_embeds = model.model.embed_tokens(ids)
        k = KeyConditionedEncryptor.sample_key(ids.shape[0], key_dim, device=ids.device, dtype=clean_embeds.dtype)
        enc_embeds = encryptor(clean_embeds, k)

        clean_logits = _forward_from_embeds(model, clean_embeds, mask)
        enc_logits = _forward_from_embeds(model, enc_embeds, mask)

        clean_lp = F.log_softmax(clean_logits.float(), dim=-1)
        enc_lp = F.log_softmax(enc_logits.float(), dim=-1)
        kl = F.kl_div(enc_lp, clean_lp.exp(), reduction="none").sum(dim=-1)

        m = mask[:, :kl.shape[1]].float()
        total_kl += (kl * m).sum().item()
        total_tokens += m.sum().item()

    mean_kl = total_kl / max(total_tokens, 1)
    encryptor.train()
    return mean_kl


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else PROJ_DIR / "key_search_bypass" / "outputs" / "encryptor" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    import wandb
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "key-search-bypasses-encrypted-activation-monitors"),
        name=f"encryptor_s{args.seed}",
        config=vars(args),
        reinit=True,
    )

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    hidden_dim = model.config.hidden_size
    print(f"Model loaded. hidden_dim={hidden_dim}")

    print("Loading data...")
    train_enc, val_enc = load_data(tokenizer, args.train_size, args.val_size, args.max_length)
    train_ids = train_enc["input_ids"]
    train_mask = train_enc["attention_mask"]
    val_ids = val_enc["input_ids"]
    val_mask = val_enc["attention_mask"]
    print(f"Train: {train_ids.shape[0]} samples, Val: {val_ids.shape[0]} samples")

    train_dataset = TensorDataset(train_ids, train_mask)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    encryptor = KeyConditionedEncryptor(hidden_dim=hidden_dim, key_dim=args.key_dim)
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        encryptor.load_state_dict(ckpt["encryptor_state_dict"])
    encryptor = encryptor.to(dtype=torch.bfloat16, device="cuda")
    print(f"Encryptor params: {sum(p.numel() for p in encryptor.parameters()):,}")

    optimizer = torch.optim.AdamW(encryptor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_lr = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    curriculum = UtilityGatedScheduler(
        lambda1_base=args.lambda1,
        lambda2_base=args.lambda2,
        warmup_steps=args.curriculum_warmup,
        tau_low=args.tau_low,
        tau_high=args.tau_high,
    )

    best_kl = float("inf")
    best_step = 0
    global_step = 0

    print(f"Starting training for {args.max_steps} optimizer steps (grad_accum={args.grad_accum})...")
    encryptor.train()
    optimizer.zero_grad()

    data_iter = iter(train_loader)
    micro_step = 0

    while global_step < args.max_steps:
        accum_loss = 0.0
        accum_util = 0.0
        accum_priv = 0.0
        accum_div = 0.0
        last_lam1, last_lam2, last_wt, last_ws = 0, 0, 0, 0

        for _ in range(args.grad_accum):
            try:
                batch_ids, batch_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch_ids, batch_mask = next(data_iter)

            batch_ids = batch_ids.cuda()
            batch_mask = batch_mask.cuda()

            with torch.no_grad():
                clean_embeds = model.model.embed_tokens(batch_ids)

            k1 = KeyConditionedEncryptor.sample_key(batch_ids.shape[0], args.key_dim, device="cuda", dtype=clean_embeds.dtype)
            k2 = KeyConditionedEncryptor.sample_key(batch_ids.shape[0], args.key_dim, device="cuda", dtype=clean_embeds.dtype)

            z1 = encryptor(clean_embeds, k1)
            z2 = encryptor(clean_embeds, k2)

            l_util = utility_loss(model, clean_embeds, z1, batch_mask)
            l_priv = privacy_loss(clean_embeds, z1, batch_mask)
            l_div = diversity_loss(z1, z2, batch_mask)

            last_lam1, last_lam2, last_wt, last_ws = curriculum.get_weights(l_util.item())
            loss = l_util + last_lam1 * l_priv + last_lam2 * l_div
            loss = loss / args.grad_accum
            loss.backward()

            accum_loss += loss.item() * args.grad_accum
            accum_util += l_util.item()
            accum_priv += l_priv.item()
            accum_div += l_div.item()
            micro_step += 1

        torch.nn.utils.clip_grad_norm_(encryptor.parameters(), 1.0)
        optimizer.step()
        scheduler_lr.step()
        optimizer.zero_grad()
        global_step += 1
        curriculum.advance()

        avg_loss = accum_loss / args.grad_accum
        avg_util = accum_util / args.grad_accum
        avg_priv = accum_priv / args.grad_accum
        avg_div = accum_div / args.grad_accum

        log_dict = {
            "train/loss": avg_loss,
            "train/util_loss": avg_util,
            "train/priv_loss": avg_priv,
            "train/div_loss": avg_div,
            "train/lambda1_eff": last_lam1,
            "train/lambda2_eff": last_lam2,
            "train/w_time": last_wt,
            "train/w_safe": last_ws,
            "train/lr": optimizer.param_groups[0]["lr"],
            "step": global_step,
        }
        wandb.log(log_dict, step=global_step)

        if global_step % 50 == 0 or global_step == 1:
            print(f"Step {global_step}/{args.max_steps} | "
                  f"loss={avg_loss:.4f} util={avg_util:.4f} priv={avg_priv:.4f} div={avg_div:.4f} | "
                  f"lam1={last_lam1:.3f} lam2={last_lam2:.3f} lr={optimizer.param_groups[0]['lr']:.6f}")

        if math.isnan(avg_loss) or avg_loss > 100:
            print(f"Training collapsed at step {global_step}: loss={avg_loss}")
            wandb.finish()
            sys.exit(1)

        min_save_step = max(200, args.warmup_steps // 2)
        if global_step % args.eval_steps == 0 or global_step == 1:
            print(f"Evaluating at step {global_step}...")
            eval_kl = evaluate(encryptor, model, val_ids, val_mask, args.eval_samples, args.key_dim)
            print(f"  Eval KL: {eval_kl:.6f} (best: {best_kl:.6f})")
            wandb.log({"eval/kl_div": eval_kl, "step": global_step}, step=global_step)

            if eval_kl < best_kl and global_step >= min_save_step:
                best_kl = eval_kl
                best_step = global_step
                ckpt = {
                    "encryptor_state_dict": encryptor.state_dict(),
                    "step": global_step,
                    "eval_kl": eval_kl,
                    "seed": args.seed,
                    "config": vars(args),
                }
                torch.save(ckpt, output_dir / "best_checkpoint.pt")
                print(f"  Saved best checkpoint (KL={eval_kl:.6f}) at step {global_step}")

    final_kl = evaluate(encryptor, model, val_ids, val_mask, args.eval_samples, args.key_dim)
    print(f"\nTraining complete. Final eval KL: {final_kl:.6f}, Best KL: {best_kl:.6f} at step {best_step}")

    ckpt_last = {
        "encryptor_state_dict": encryptor.state_dict(),
        "step": global_step,
        "eval_kl": final_kl,
        "seed": args.seed,
        "config": vars(args),
    }
    torch.save(ckpt_last, output_dir / "last_checkpoint.pt")

    if best_step == 0:
        print("No best checkpoint was saved during training (min_save_step not reached). Using last checkpoint.")
        best_kl = final_kl
        best_step = global_step
        torch.save(ckpt_last, output_dir / "best_checkpoint.pt")

    summary = {
        "seed": args.seed,
        "best_step": best_step,
        "best_kl": best_kl,
        "final_kl": final_kl,
        "total_steps": global_step,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training summary to {output_dir / 'training_summary.json'}")

    wandb.log({"best_kl": best_kl, "best_step": best_step})
    wandb.finish()


if __name__ == "__main__":
    main()
