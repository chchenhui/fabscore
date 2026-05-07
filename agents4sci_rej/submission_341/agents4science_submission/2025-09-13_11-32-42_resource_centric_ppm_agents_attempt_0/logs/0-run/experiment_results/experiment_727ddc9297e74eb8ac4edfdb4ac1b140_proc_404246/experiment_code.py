import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
import random
import math
import matplotlib.pyplot as plt
from collections import defaultdict

# Device handling (required)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experiment data structure
experiment_data = {
    "BPI2012": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "resource_metrics": {"wMAPE_H": []},
    },
    "BPI2017": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "resource_metrics": {"wMAPE_H": []},
    },
    "ROAD": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "resource_metrics": {"wMAPE_H": []},
    },
}

# Data loading utilities
from ai_scientist.ideas.my_research_topic import load_datasets, pick_default_dataset


# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# Prefix dataset construction
def build_prefix_dataset(df, max_prefix_len=10, min_prefix_len=1):
    df = df.copy()
    if "lifecycle" in df.columns:
        lc = df["lifecycle"].astype(str).str.lower()
        if (lc == "complete").any():
            df = df[lc == "complete"]
    df = df.sort_values(["case_id", "timestamp"])
    # vocab
    acts = sorted(df["activity"].astype(str).unique().tolist())
    act2id = {a: i + 1 for i, a in enumerate(acts)}  # 0 PAD
    id2act = {i: a for a, i in act2id.items()}
    pad_id = 0

    samples = []
    for cid, g in df.groupby("case_id"):
        g = g.sort_values("timestamp")
        if len(g) < 2:
            continue
        ts = (
            pd.to_datetime(g["timestamp"], utc=True).astype("int64").to_numpy() // 10**9
        )
        acts_ids = np.array(
            [act2id[a] for a in g["activity"].astype(str)], dtype=np.int64
        )
        gts = pd.to_datetime(g["timestamp"], utc=True)
        hours = (gts.dt.hour.to_numpy(dtype=float) / 23.0).astype(np.float32)
        weekdays = (gts.dt.weekday.to_numpy(dtype=float) / 6.0).astype(np.float32)
        working = (
            ((gts.dt.weekday < 5) & (gts.dt.hour >= 8) & (gts.dt.hour <= 17)).astype(
                float
            )
        ).astype(np.float32)
        deltas = np.diff(ts, prepend=ts[0]).astype(np.float32)
        since_start = (ts - ts[0]).astype(np.float32)
        feats = np.stack(
            [deltas, since_start, hours, weekdays, working], axis=1
        ).astype(np.float32)
        T = len(acts_ids)
        max_k = min(max_prefix_len, T - 1)
        for k in range(min_prefix_len, max_k + 1):
            samples.append(
                {
                    "case_id": cid,
                    "seq_acts": acts_ids[:k].tolist(),
                    "seq_feats": feats[:k].copy(),
                    "target": int(acts_ids[k]),
                    "anchor_ts": int(ts[k - 1]),
                    "next_ts": int(ts[k]),
                }
            )
    # initial normalization (will redo on train)
    if len(samples) == 0:
        return samples, act2id, id2act, pad_id
    all_feats = np.concatenate(
        [s["seq_feats"] for s in samples if len(s["seq_feats"]) > 0], axis=0
    )
    for s in samples:
        s["seq_feats"][:, 0] = (s["seq_feats"][:, 0] - all_feats[:, 0].mean()) / (
            all_feats[:, 0].std() + 1e-6
        )
        s["seq_feats"][:, 1] = (s["seq_feats"][:, 1] - all_feats[:, 1].mean()) / (
            all_feats[:, 1].std() + 1e-6
        )
    return samples, act2id, id2act, pad_id


# Time-based split by case start
def time_based_split(df, train_frac=0.7, val_frac=0.15):
    starts = (
        df.sort_values("timestamp").groupby("case_id")["timestamp"].min().reset_index()
    )
    starts = starts.sort_values("timestamp").reset_index(drop=True)
    n = len(starts)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_cases = set(starts.iloc[:n_train]["case_id"])
    val_cases = set(starts.iloc[n_train : n_train + n_val]["case_id"])
    test_cases = set(starts.iloc[n_train + n_val :]["case_id"])
    return train_cases, val_cases, test_cases


class PrefixDataset(Dataset):
    def __init__(self, samples, pad_id, max_len=10, num_cont=5):
        self.samples = samples
        self.pad_id = pad_id
        self.max_len = max_len
        self.num_cont = num_cont

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        seq = s["seq_acts"][-self.max_len :]
        feats = s["seq_feats"][-self.max_len :]
        L = len(seq)
        pad_len = self.max_len - L
        seq_pad = [self.pad_id] * pad_len + seq
        feats_pad = np.zeros((pad_len, self.num_cont), dtype=np.float32)
        feats_pad = np.vstack([feats_pad, feats.astype(np.float32)])
        attn_mask = np.array([0] * pad_len + [1] * L, dtype=np.float32)
        return {
            "acts": torch.tensor(seq_pad, dtype=torch.long),
            "feats": torch.tensor(feats_pad, dtype=torch.float32),
            "mask": torch.tensor(attn_mask, dtype=torch.float32),
            "y": torch.tensor(s["target"], dtype=torch.long),
        }


class LSTMBaseline(nn.Module):
    def __init__(
        self, vocab_size, emb_dim=64, cont_dim=5, hidden=128, num_layers=1, pad_idx=0
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim + cont_dim,
            hidden_size=hidden,
            batch_first=True,
            num_layers=num_layers,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden, vocab_size + 1)
        self.pad_idx = pad_idx

    def forward(self, acts, feats, mask):
        x = self.emb(acts)
        x = torch.cat([x, feats], dim=-1)
        out, (h, c) = self.lstm(x)
        h_last = h[-1]
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits


def collate_fn(batch):
    keys = batch[0].keys()
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    return out


def evaluate(model, loader, criterion, device, num_classes, pad_idx):
    model.eval()
    total_loss, n_total = 0.0, 0
    ys, preds_top1, preds_probs = [], [], []
    top3_correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["acts"], batch["feats"], batch["mask"])
            loss = criterion(logits, batch["y"])
            total_loss += loss.item() * logits.size(0)
            probs = torch.softmax(logits, dim=1)
            top1 = torch.argmax(probs, dim=1)
            k_val = min(3, probs.size(1))
            _, topk_idx = torch.topk(probs, k=k_val, dim=1)
            ys.extend(batch["y"].detach().cpu().tolist())
            preds_top1.extend(top1.detach().cpu().tolist())
            preds_probs.append(probs.detach().cpu().numpy())
            for i in range(batch["y"].size(0)):
                if batch["y"][i].item() in topk_idx[i].detach().cpu().tolist():
                    top3_correct += 1
            n_total += batch["y"].size(0)
    avg_loss = total_loss / max(1, n_total)
    y_true = np.array(ys)
    y_pred = np.array(preds_top1)
    mask = y_true != pad_idx
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    try:
        f1 = float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        f1 = 0.0
    top3 = float(top3_correct / max(1, n_total))
    probs_concat = (
        np.concatenate(preds_probs, axis=0)
        if len(preds_probs) > 0
        else np.zeros((0, num_classes + 1))
    )
    return avg_loss, acc, f1, top3, y_true, y_pred, probs_concat


# Resource-centric simulator utilities
def tod_bin(hour, nbins=6):
    # 6 bins of 4 hours: 0-3,4-7,...,20-23
    return int(hour * nbins // 24)


def estimate_resource_rates(train_df, nbins=6):
    if "resource" not in train_df.columns:
        return {}, 0.0
    df = train_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = df["timestamp"].dt.hour
    df["bin"] = df["hour"].apply(lambda h: tod_bin(h, nbins))
    # time exposure per bin (hours) over observed days
    tmin = df["timestamp"].min()
    tmax = df["timestamp"].max()
    # guard
    if pd.isna(tmin) or pd.isna(tmax):
        return {}, 0.0
    total_days = max(1.0, (tmax - tmin).total_seconds() / (3600 * 24))
    bin_hours = (24.0 / nbins) * total_days
    # counts
    rates = defaultdict(lambda: np.zeros(nbins, dtype=float))
    for (r, b), g in df.groupby(["resource", "bin"]):
        rates[r][b] = len(g) / bin_hours  # completions per hour
    return rates, bin_hours


def simulate_workload_MC(anchor_times, rates, H_seconds=3600, nbins=6, N=30):
    # For each anchor time, simulate per-resource completions in next H
    resources = list(rates.keys())
    if len(resources) == 0:
        return [], resources
    H_hours = H_seconds / 3600.0
    sims = []
    for t0 in anchor_times:
        t0 = pd.to_datetime(pd.Timestamp.utcfromtimestamp(int(t0)), utc=True)
        hr = t0.hour
        b = tod_bin(hr, nbins)
        mu_r = np.array([rates[r][b] * H_hours for r in resources], dtype=float)
        # Monte Carlo Poisson draws
        draws = np.random.poisson(lam=mu_r[None, :], size=(N, len(resources)))
        sims.append(
            draws.mean(axis=0)
        )  # expected completions per resource over horizon
    return sims, resources


def realized_workload_counts(test_df, anchor_times, resources, H_seconds=3600):
    if len(resources) == 0:
        return []
    df = test_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    ts = df["timestamp"].to_numpy()
    res = (
        df["resource"].astype(str).to_numpy()
        if "resource" in df.columns
        else np.array(["System"] * len(df))
    )
    realized = []
    for t0 in anchor_times:
        t0pd = pd.to_datetime(pd.Timestamp.utcfromtimestamp(int(t0)), utc=True)
        t1pd = t0pd + pd.Timedelta(seconds=H_seconds)
        mask = (ts > t0pd) & (ts <= t1pd)
        counts = np.zeros(len(resources), dtype=float)
        if mask.any():
            res_slice = res[mask]
            for i, r in enumerate(resources):
                counts[i] = np.sum(res_slice == r)
        realized.append(counts)
    return realized


def compute_rw_wmape(sim_list, real_list):
    # Weighted MAPE across anchors and resources: sum |sim-real| weighted by real over sum(real)
    eps = 1e-6
    num = 0.0
    den = 0.0
    for sim, real in zip(sim_list, real_list):
        sim = np.array(sim, dtype=float)
        real = np.array(real, dtype=float)
        num += np.sum(np.abs(sim - real) * real)
        den += np.sum(real)
    return float(num / max(den, eps))


def train_one_dataset(
    name,
    df,
    max_epochs=6,
    batch_size=128,
    max_prefix_len=10,
    lr=1e-3,
    H_seconds=3600,
    nbins=6,
    N_mc=30,
):
    print(f"\n=== Dataset: {name} ===")
    # Time-based split by case start
    train_cases, val_cases, test_cases = time_based_split(df, 0.7, 0.15)
    # Build all samples and vocab
    samples_all, act2id, id2act, pad_id = build_prefix_dataset(
        df, max_prefix_len=max_prefix_len
    )
    samples_train = [s for s in samples_all if s["case_id"] in train_cases]
    samples_val = [s for s in samples_all if s["case_id"] in val_cases]
    samples_test = [s for s in samples_all if s["case_id"] in test_cases]
    # Re-normalize using train only
    if len(samples_train) > 0:
        feats_train = [
            s["seq_feats"] for s in samples_train if s["seq_feats"].shape[0] > 0
        ]
        if len(feats_train) > 0:
            all_feats = np.concatenate(feats_train, axis=0)
            dt_mean, dt_std = all_feats[:, 0].mean(), all_feats[:, 0].std() + 1e-6
            ss_mean, ss_std = all_feats[:, 1].mean(), all_feats[:, 1].std() + 1e-6
            for S in (samples_train, samples_val, samples_test):
                for s in S:
                    s["seq_feats"][:, 0] = (s["seq_feats"][:, 0] - dt_mean) / dt_std
                    s["seq_feats"][:, 1] = (s["seq_feats"][:, 1] - ss_mean) / ss_std
    print(
        f"Samples train/val/test: {len(samples_train)}/{len(samples_val)}/{len(samples_test)}; vocab={len(act2id)}"
    )
    if len(samples_train) == 0 or len(act2id) < 2:
        print("Not enough data; skipping.")
        return
    ds_train = PrefixDataset(
        samples_train, pad_id=pad_id, max_len=max_prefix_len, num_cont=5
    )
    ds_val = PrefixDataset(
        samples_val, pad_id=pad_id, max_len=max_prefix_len, num_cont=5
    )
    ds_test = PrefixDataset(
        samples_test, pad_id=pad_id, max_len=max_prefix_len, num_cont=5
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Prepare resource simulator from train split
    df_train = df[df["case_id"].isin(train_cases)].copy()
    df_val = df[df["case_id"].isin(val_cases)].copy()
    df_test = df[df["case_id"].isin(test_cases)].copy()
    rates, bin_hours = estimate_resource_rates(df_train, nbins=nbins)
    # Anchor times from test prefixes (subsample for speed)
    anchor_times = [
        s["anchor_ts"] for s in samples_test[:: max(1, len(samples_test) // 2000 or 1)]
    ]
    sim_list, resources = simulate_workload_MC(
        anchor_times, rates, H_seconds=H_seconds, nbins=nbins, N=N_mc
    )
    real_list = realized_workload_counts(
        df_test, anchor_times, resources, H_seconds=H_seconds
    )
    rw_wmape_baseline = (
        compute_rw_wmape(sim_list, real_list) if len(resources) > 0 else None
    )
    if rw_wmape_baseline is not None:
        print(
            f"[{name}] Initial Resource-Weighted wMAPE@H={rw_wmape_baseline:.4f} (H={H_seconds/60:.0f}m, resources={len(resources)})"
        )
    else:
        print(f"[{name}] Resource data missing; skipping wMAPE@H.")

    # Model
    model = LSTMBaseline(
        vocab_size=len(act2id),
        emb_dim=64,
        cont_dim=5,
        hidden=128,
        num_layers=1,
        pad_idx=pad_id,
    ).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    best_val_top3 = -1.0
    best_state = None
    hist = {"train_loss": [], "val_loss": [], "val_top3": []}
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, running_loss = 0, 0.0
        for batch in dl_train:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            optimizer.zero_grad()
            logits = model(batch["acts"], batch["feats"], batch["mask"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * logits.size(0)
            total += logits.size(0)
        train_loss = running_loss / max(1, total)
        val_loss, val_acc, val_f1, val_top3, _, _, _ = evaluate(
            model, dl_val, criterion, device, len(act2id), pad_id
        )
        print(
            f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | val_top3={val_top3:.4f}"
        )
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["val_top3"].append(val_top3)
        experiment_data[name]["losses"]["train"].append((epoch, train_loss))
        experiment_data[name]["losses"]["val"].append((epoch, val_loss))
        experiment_data[name]["metrics"]["val"].append(
            (epoch, {"acc": val_acc, "macro_f1": val_f1, "top3": val_top3})
        )
        experiment_data[name]["epochs"].append(epoch)
        # Update resource metric (static in this simple baseline but tracked per epoch)
        if rw_wmape_baseline is not None:
            experiment_data[name]["resource_metrics"]["wMAPE_H"].append(
                (epoch, rw_wmape_baseline)
            )
        if val_top3 > best_val_top3:
            best_val_top3 = val_top3
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Final eval
    train_loss, train_acc, train_f1, train_top3, _, _, _ = evaluate(
        model, dl_train, criterion, device, len(act2id), pad_id
    )
    val_loss, val_acc, val_f1, val_top3, _, _, _ = evaluate(
        model, dl_val, criterion, device, len(act2id), pad_id
    )
    test_loss, test_acc, test_f1, test_top3, y_true_t, y_pred_t, probs_t = evaluate(
        model, dl_test, criterion, device, len(act2id), pad_id
    )
    print(
        f"[{name}] Train: loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} top3={train_top3:.4f}"
    )
    print(
        f"[{name}] Test:  loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f} top3={test_top3:.4f}"
    )

    # Save metrics
    experiment_data[name]["metrics"]["train"].append(
        (
            "final",
            {
                "loss": train_loss,
                "acc": train_acc,
                "macro_f1": train_f1,
                "top3": train_top3,
            },
        )
    )
    experiment_data[name]["metrics"]["val"].append(
        (
            "final",
            {"loss": val_loss, "acc": val_acc, "macro_f1": val_f1, "top3": val_top3},
        )
    )
    experiment_data[name]["metrics"]["test"].append(
        (
            "final",
            {
                "loss": test_loss,
                "acc": test_acc,
                "macro_f1": test_f1,
                "top3": test_top3,
            },
        )
    )
    experiment_data[name]["predictions"] = y_pred_t.tolist()
    experiment_data[name]["ground_truth"] = y_true_t.tolist()

    # Plots
    try:
        plt.figure()
        plt.plot(hist["train_loss"], label="train_loss")
        plt.plot(hist["val_loss"], label="val_loss")
        plt.legend()
        plt.title(f"Loss Curves - {name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{name}.png"))
        plt.close()
        plt.figure()
        plt.plot(hist["val_top3"], label="val_top3")
        plt.legend()
        plt.title(f"Val Top-3 Acc - {name}")
        plt.xlabel("Epoch")
        plt.ylabel("Top-3 Acc")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"val_top3_{name}.png"))
        plt.close()
        if len(experiment_data[name]["resource_metrics"]["wMAPE_H"]) > 0:
            xs = [e for e, _ in experiment_data[name]["resource_metrics"]["wMAPE_H"]]
            ys = [m for _, m in experiment_data[name]["resource_metrics"]["wMAPE_H"]]
            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.title(f"Resource wMAPE@H - {name}")
            plt.xlabel("Epoch")
            plt.ylabel("wMAPE")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"rw_wmape_{name}.png"))
            plt.close()
    except Exception as e:
        print(f"[warn] Plotting failed: {e}")

    # Confusion matrix
    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true_t, y_pred_t)
        np.save(os.path.join(working_dir, f"cm_{name}.npy"), cm)
    except Exception as e:
        print(f"[warn] Confusion matrix failed: {e}")


def main():
    datasets = load_datasets()
    # Loop datasets; cap to earliest 5000 cases
    for key, df in datasets.items():
        try:
            starts = (
                df.sort_values("timestamp")
                .groupby("case_id")["timestamp"]
                .min()
                .reset_index()
            )
            if len(starts) > 5000:
                keep_cases = set(starts.iloc[:5000]["case_id"])
                df_small = df[df["case_id"].isin(keep_cases)].copy()
            else:
                df_small = df
        except Exception:
            df_small = df
        train_one_dataset(
            key,
            df_small,
            max_epochs=6,
            batch_size=128,
            max_prefix_len=10,
            lr=1e-3,
            H_seconds=3600,
            nbins=6,
            N_mc=30,
        )
    # Save experiment data
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    np.savez_compressed(
        os.path.join(working_dir, "experiment_data_compressed.npz"),
        data=experiment_data,
    )
    # Print summary
    for k in experiment_data.keys():
        m = (
            experiment_data[k]["metrics"]["test"][-1][1]
            if len(experiment_data[k]["metrics"]["test"]) > 0
            else {}
        )
        rw = (
            experiment_data[k]["resource_metrics"]["wMAPE_H"][-1][1]
            if len(experiment_data[k]["resource_metrics"]["wMAPE_H"]) > 0
            else None
        )
        print(
            f"[Summary {k}] Test acc={m.get('acc', None)} macro_f1={m.get('macro_f1', None)} top3={m.get('top3', None)} wMAPE@H={rw}"
        )


# Execute immediately
main()
