import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import random
import matplotlib.pyplot as plt
from ai_scientist.ideas.my_research_topic import load_datasets

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
        "wMAPE_H60": {"val": [], "test": []},
    },
    "BPI2017": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "wMAPE_H60": {"val": [], "test": []},
    },
    "ROAD": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "wMAPE_H60": {"val": [], "test": []},
    },
}


# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# Prefix building
def build_prefix_dataset(df, max_prefix_len=10, min_prefix_len=1):
    df = df.copy()
    if "lifecycle" in df.columns:
        df = df[df["lifecycle"].astype(str).str.lower().eq("complete")]
        if len(df) == 0:
            df = df.sort_values(["case_id", "timestamp"])
    df = df.sort_values(["case_id", "timestamp"])
    acts = df["activity"].astype(str).unique().tolist()
    act2id = {a: i + 1 for i, a in enumerate(sorted(acts))}
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
            [act2id[a] for a in g["activity"].astype(str).tolist()], dtype=np.int64
        )
        g_ts = pd.to_datetime(g["timestamp"], utc=True)
        hours = (g_ts.dt.hour.to_numpy(dtype=float) / 23.0).astype(np.float32)
        weekdays = (g_ts.dt.weekday.to_numpy(dtype=float) / 6.0).astype(np.float32)
        working = (
            (g_ts.dt.weekday.to_numpy() < 5)
            & (g_ts.dt.hour.to_numpy() >= 8)
            & (g_ts.dt.hour.to_numpy() <= 17)
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
                    "last_ts": int(ts[k - 1]),
                    "next_ts": int(ts[k]),
                }
            )
    if len(samples) == 0:
        return [], act2id, id2act, pad_id
    all_feats = np.concatenate(
        [s["seq_feats"] for s in samples if len(s["seq_feats"]) > 0], axis=0
    )
    dt_mean, dt_std = all_feats[:, 0].mean(), all_feats[:, 0].std() + 1e-6
    ss_mean, ss_std = all_feats[:, 1].mean(), all_feats[:, 1].std() + 1e-6
    for s in samples:
        if s["seq_feats"].shape[0] > 0:
            s["seq_feats"][:, 0] = (s["seq_feats"][:, 0] - dt_mean) / dt_std
            s["seq_feats"][:, 1] = (s["seq_feats"][:, 1] - ss_mean) / ss_std
    return samples, act2id, id2act, pad_id


# Time-based split
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
        h_last = self.dropout(h[-1])
        logits = self.fc(h_last)
        return logits


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def evaluate(model, loader, criterion, device, num_classes, pad_idx):
    model.eval()
    total_loss = 0.0
    ys, preds_top1, preds_probs = [], [], []
    top3_correct = 0
    n_total = 0
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


# Resource-centric workload estimation utils
def compute_resource_rates(train_df):
    df = train_df.copy()
    if "lifecycle" in df.columns:
        df = df[df["lifecycle"].astype(str).str.lower().eq("complete")]
    if len(df) == 0:
        return {}
    t0 = pd.to_datetime(df["timestamp"], utc=True).min()
    t1 = pd.to_datetime(df["timestamp"], utc=True).max()
    minutes = max(1.0, (t1 - t0).total_seconds() / 60.0)
    counts = df.groupby("resource").size().to_dict()
    rates = {r: c / minutes for r, c in counts.items()}
    return rates


def prepare_future_index(df_cases):
    df = df_cases.copy()
    if "lifecycle" in df.columns:
        df = df[df["lifecycle"].astype(str).str.lower().eq("complete")]
    df["ts_sec"] = pd.to_datetime(df["timestamp"], utc=True).astype("int64") // 10**9
    # Build per-resource sorted timestamp arrays
    res_ts = {}
    for r, g in df.groupby("resource"):
        arr = np.sort(g["ts_sec"].to_numpy(dtype=np.int64))
        res_ts[r] = arr
    return res_ts


def count_events_in_interval(ts_array, start_s, end_s):
    # Count events with (start_s, end_s] using binary search
    import bisect

    right = bisect.bisect_right(ts_array, end_s)
    left = bisect.bisect_right(ts_array, start_s)
    return max(0, right - left)


def resource_wmape_at_H(
    prefixes,
    res_ts_val,
    res_rates,
    H_minutes=60,
    max_prefix_eval=1000,
    monte_carlo_runs=10,
):
    if len(prefixes) == 0:
        return 0.0
    idxs = np.random.choice(
        len(prefixes), size=min(max_prefix_eval, len(prefixes)), replace=False
    )
    total_weighted_abs_perc = 0.0
    total_weight = 0.0
    for i in idxs:
        t0 = int(prefixes[i]["last_ts"])
        t1 = t0 + int(H_minutes * 60)
        # actual per resource
        for r, ts_arr in res_ts_val.items():
            actual = count_events_in_interval(ts_arr, t0, t1)
            # Monte Carlo forecast: Poisson with rate lambda*H
            lam = res_rates.get(r, 0.0) * H_minutes
            # Average of MC samples approximates lam
            if monte_carlo_runs > 0 and lam > 0:
                sims = np.random.poisson(lam=lam, size=monte_carlo_runs)
                forecast = float(np.mean(sims))
            else:
                forecast = lam
            w = float(actual)
            if actual == 0:
                # still include small epsilon weight to avoid division by zero but not dominate
                continue
            total_weighted_abs_perc += w * (abs(forecast - actual) / max(1e-6, actual))
            total_weight += w
    if total_weight == 0:
        return 0.0
    return float(total_weighted_abs_perc / total_weight)


def train_one_dataset(
    name, df, max_epochs=8, batch_size=128, max_prefix_len=10, lr=1e-3, H_minutes=60
):
    print(f"\n=== Dataset: {name} ===")
    # Cap to earliest 5000 cases for speed
    try:
        starts_all = (
            df.sort_values("timestamp")
            .groupby("case_id")["timestamp"]
            .min()
            .reset_index()
        )
        if len(starts_all) > 5000:
            keep_cases = set(starts_all.iloc[:5000]["case_id"])
            df = df[df["case_id"].isin(keep_cases)].copy()
    except Exception:
        pass

    # Splits
    train_cases, val_cases, test_cases = time_based_split(df, 0.7, 0.15)
    df_train = df[df["case_id"].isin(train_cases)].copy()
    df_val = df[df["case_id"].isin(val_cases)].copy()
    df_test = df[df["case_id"].isin(test_cases)].copy()

    # Build samples
    samples_all, act2id, id2act, pad_id = build_prefix_dataset(
        df, max_prefix_len=max_prefix_len
    )
    samples_train = [s for s in samples_all if s["case_id"] in train_cases]
    samples_val = [s for s in samples_all if s["case_id"] in val_cases]
    samples_test = [s for s in samples_all if s["case_id"] in test_cases]

    # Re-normalize using train statistics (already normalized globally; recompute on train)
    if len(samples_train) > 0:
        concat_feats = [
            s["seq_feats"] for s in samples_train if s["seq_feats"].shape[0] > 0
        ]
        if len(concat_feats) > 0:
            all_feats = np.concatenate(concat_feats, axis=0)
            dt_mean, dt_std = all_feats[:, 0].mean(), all_feats[:, 0].std() + 1e-6
            ss_mean, ss_std = all_feats[:, 1].mean(), all_feats[:, 1].std() + 1e-6

            def renorm(samples):
                for s in samples:
                    if s["seq_feats"].shape[0] > 0:
                        s["seq_feats"][:, 0] = (s["seq_feats"][:, 0] - dt_mean) / dt_std
                        s["seq_feats"][:, 1] = (s["seq_feats"][:, 1] - ss_mean) / ss_std

            renorm(samples_train)
            renorm(samples_val)
            renorm(samples_test)

    print(
        f"Samples train/val/test: {len(samples_train)}/{len(samples_val)}/{len(samples_test)}; vocab={len(act2id)}"
    )
    if len(samples_train) == 0 or len(act2id) < 2:
        print("Not enough data to train. Skipping.")
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

    # Resource rate and future index for resource-centric metric
    res_rates = compute_resource_rates(df_train)
    res_ts_val = prepare_future_index(df_val)
    res_ts_test = prepare_future_index(df_test)

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

    best_val_top3 = -1.0
    best_state = None
    hist = {"train_loss": [], "val_loss": [], "val_top3": [], "val_wmape": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0
        running_loss = 0.0
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
        # Resource-Weighted Workload wMAPE@H on validation
        wmape_val = resource_wmape_at_H(
            samples_val,
            res_ts_val,
            res_rates,
            H_minutes=H_minutes,
            max_prefix_eval=1000,
            monte_carlo_runs=10,
        )

        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
        print(
            f"Epoch {epoch}: val_acc={val_acc:.4f} | val_macroF1={val_f1:.4f} | val_top3={val_top3:.4f} | wMAPE@H={wmape_val:.4f}"
        )

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["val_top3"].append(val_top3)
        hist["val_wmape"].append(wmape_val)

        experiment_data[name]["losses"]["train"].append((epoch, train_loss))
        experiment_data[name]["losses"]["val"].append((epoch, val_loss))
        experiment_data[name]["metrics"]["val"].append(
            (
                epoch,
                {
                    "acc": val_acc,
                    "macro_f1": val_f1,
                    "top3": val_top3,
                    "wMAPE_H60": wmape_val,
                },
            )
        )
        experiment_data[name]["epochs"].append(epoch)

        if val_top3 > best_val_top3:
            best_val_top3 = val_top3
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Final evaluations
    train_loss, train_acc, train_f1, train_top3, _, _, _ = evaluate(
        model, dl_train, criterion, device, len(act2id), pad_id
    )
    val_loss, val_acc, val_f1, val_top3, _, _, _ = evaluate(
        model, dl_val, criterion, device, len(act2id), pad_id
    )
    test_loss, test_acc, test_f1, test_top3, y_true_t, y_pred_t, probs_t = evaluate(
        model, dl_test, criterion, device, len(act2id), pad_id
    )

    # Final resource-centric metrics on test
    wmape_test = resource_wmape_at_H(
        samples_test,
        res_ts_test,
        res_rates,
        H_minutes=H_minutes,
        max_prefix_eval=2000,
        monte_carlo_runs=10,
    )

    print(
        f"[{name}] Train: loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} top3={train_top3:.4f}"
    )
    print(
        f"[{name}] Val:   loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} top3={val_top3:.4f}"
    )
    print(
        f"[{name}] Test:  loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f} top3={test_top3:.4f} wMAPE@H={wmape_test:.4f}"
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
                "wMAPE_H60": wmape_test,
            },
        )
    )
    experiment_data[name]["predictions"] = y_pred_t.tolist()
    experiment_data[name]["ground_truth"] = y_true_t.tolist()
    experiment_data[name]["wMAPE_H60"]["val"].append(
        ("final", hist["val_wmape"][-1] if len(hist["val_wmape"]) > 0 else 0.0)
    )
    experiment_data[name]["wMAPE_H60"]["test"].append(("final", wmape_test))

    # Save plots
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

        plt.figure()
        plt.plot(hist["val_wmape"], label="val_wMAPE@H60")
        plt.legend()
        plt.title(f"Val wMAPE@H60 - {name}")
        plt.xlabel("Epoch")
        plt.ylabel("wMAPE (lower better)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"val_wmape_{name}.png"))
        plt.close()
    except Exception as e:
        print(f"[warn] Plotting failed: {e}")

    # Save confusion matrix
    try:
        cm = confusion_matrix(y_true_t, y_pred_t)
        np.save(os.path.join(working_dir, f"cm_{name}.npy"), cm)
    except Exception as e:
        print(f"[warn] CM save failed: {e}")


def run():
    datasets = load_datasets()
    for key, df in datasets.items():
        try:
            train_one_dataset(
                key,
                df,
                max_epochs=8,
                batch_size=128,
                max_prefix_len=10,
                lr=1e-3,
                H_minutes=60,
            )
        except Exception as e:
            print(f"[warn] Dataset {key} failed with error: {e}")
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    np.savez_compressed(
        os.path.join(working_dir, "experiment_data_compressed.npz"),
        data=experiment_data,
    )


# Execute immediately
run()
