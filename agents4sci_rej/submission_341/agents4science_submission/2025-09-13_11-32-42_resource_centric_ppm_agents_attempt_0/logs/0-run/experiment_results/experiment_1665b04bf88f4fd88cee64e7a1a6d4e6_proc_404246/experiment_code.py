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
from collections import defaultdict, Counter
from datetime import timedelta

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
        "resource_workload": {"pred": [], "actual": [], "wmape": []},
    },
    "BPI2017": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "resource_workload": {"pred": [], "actual": [], "wmape": []},
    },
    "ROAD": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "resource_workload": {"pred": [], "actual": [], "wmape": []},
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


# Helpers
def ensure_complete_only(df):
    if "lifecycle" in df.columns:
        mask = df["lifecycle"].astype(str).str.lower().eq("complete")
        if mask.any():
            return df[mask].copy()
    return df.copy()


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


def build_prefix_dataset(df, max_prefix_len=10, min_prefix_len=1):
    df = ensure_complete_only(df)
    df = df.sort_values(["case_id", "timestamp"])
    acts = df["activity"].astype(str).unique().tolist()
    act2id = {a: i + 1 for i, a in enumerate(sorted(acts))}
    id2act = {i: a for a, i in act2id.items()}
    pad_id = 0
    samples = []
    for cid, g in df.groupby("case_id"):
        g = g.sort_values("timestamp")
        ts = pd.to_datetime(g["timestamp"], utc=True)
        ts_s = (ts.astype("int64") // 10**9).to_numpy(dtype=np.int64)
        acts_ids = np.array(
            [act2id[a] for a in g["activity"].astype(str)], dtype=np.int64
        )
        if len(acts_ids) < 2:
            continue
        hours = (ts.dt.hour.to_numpy(dtype=float) / 23.0).astype(np.float32)
        weekdays = (ts.dt.weekday.to_numpy(dtype=float) / 6.0).astype(np.float32)
        working = (
            (
                (ts.dt.weekday.to_numpy() < 5)
                & (ts.dt.hour.to_numpy() >= 8)
                & (ts.dt.hour.to_numpy() <= 17)
            )
        ).astype(np.float32)
        deltas = np.diff(ts_s, prepend=ts_s[0]).astype(np.float32)
        since_start = (ts_s - ts_s[0]).astype(np.float32)
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
                    "last_ts": int(ts_s[k - 1]),
                    "next_ts": int(ts_s[k]),
                }
            )
    # Initial normalization (recomputed on train)
    if len(samples) > 0:
        all_feats = np.concatenate(
            [s["seq_feats"] for s in samples if len(s["seq_feats"]) > 0], axis=0
        )
        for s in samples:
            if s["seq_feats"].shape[0] > 0:
                s["seq_feats"][:, 0] = (
                    s["seq_feats"][:, 0] - all_feats[:, 0].mean()
                ) / (all_feats[:, 0].std() + 1e-6)
                s["seq_feats"][:, 1] = (
                    s["seq_feats"][:, 1] - all_feats[:, 1].mean()
                ) / (all_feats[:, 1].std() + 1e-6)
    return samples, act2id, id2act, pad_id


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
        pad = self.max_len - L
        seq_pad = [self.pad_id] * pad + seq
        feats_pad = np.zeros((pad, self.num_cont), dtype=np.float32)
        feats_pad = np.vstack([feats_pad, feats.astype(np.float32)])
        mask = np.array([0] * pad + [1] * L, dtype=np.float32)
        return {
            "acts": torch.tensor(seq_pad, dtype=torch.long),
            "feats": torch.tensor(feats_pad, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "y": torch.tensor(s["target"], dtype=torch.long),
        }


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


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


def evaluate(model, loader, criterion, device, num_classes, pad_idx):
    model.eval()
    total_loss = 0.0
    ys = []
    preds_top1 = []
    preds_probs = []
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


# Resource-centric components
def estimate_activity_durations(df):
    df = df.sort_values(["case_id", "timestamp"]).copy()
    df["next_ts"] = df.groupby("case_id")["timestamp"].shift(-1)
    df["dur"] = (df["next_ts"] - df["timestamp"]).dt.total_seconds()
    # filter unreasonable durations
    df["dur"] = df["dur"].clip(lower=60, upper=8 * 3600)  # 1min to 8h
    dura = df.groupby("activity")["dur"].median().to_dict()
    # fallback
    global_med = float(np.nanmedian(list(dura.values()))) if len(dura) > 0 else 1800.0
    dura = {k: (float(v) if not np.isnan(v) else global_med) for k, v in dura.items()}
    return dura, global_med


def tod_bin(ts):
    h = ts.hour
    if 6 <= h < 12:
        return 0
    if 12 <= h < 18:
        return 1
    if 18 <= h < 24:
        return 2
    return 3


def build_resource_policy(df):
    # Build transition counts: (resource, prev_act, tod_bin) -> next_act counts
    df = df.sort_values(["case_id", "timestamp"]).copy()
    df["next_act"] = df.groupby("case_id")["activity"].shift(-1)
    df["prev_act"] = df.groupby("case_id")["activity"].shift(0)
    df["tod"] = df["timestamp"].dt.tz_convert(None).apply(tod_bin)
    counts_r = defaultdict(Counter)
    counts_rp = defaultdict(Counter)
    counts_p = defaultdict(Counter)
    counts_global = Counter()
    for _, row in df.dropna(subset=["next_act"]).iterrows():
        r = str(row.get("resource", "System"))
        pa = str(row["prev_act"])
        na = str(row["next_act"])
        tb = int(row["tod"])
        counts_r[(r, pa, tb)][na] += 1
        counts_rp[(r, pa)][na] += 1
        counts_p[(pa, tb)][na] += 1
        counts_global[na] += 1

    def sample_next(r, prev_act, ts):
        tb = tod_bin(ts)
        for keyspace in [
            counts_r.get((r, prev_act, tb)),
            counts_rp.get((r, prev_act)),
            counts_p.get((prev_act, tb)),
            counts_global,
        ]:
            if keyspace and len(keyspace) > 0:
                acts = list(keyspace.keys())
                vals = np.array([keyspace[a] for a in acts], dtype=float)
                probs = vals / vals.sum()
                return np.random.choice(acts, p=probs)
        return None

    # compute resource idle estimates (mean inter-event gaps per resource)
    df = df.sort_values(["resource", "timestamp"])
    gaps = df.groupby("resource")["timestamp"].diff().dt.total_seconds()
    idle = gaps.clip(lower=0, upper=4 * 3600).groupby(df["resource"]).median().to_dict()
    global_idle = float(
        np.nanmedian([v for v in idle.values() if not np.isnan(v)])
        if len(idle) > 0
        else 300.0
    )
    idle = {
        k: (float(v) if not (v is None or np.isnan(v)) else global_idle)
        for k, v in idle.items()
    }
    return sample_next, idle


def simulate_resource_workload(
    resources,
    start_time,
    horizon_s,
    bin_s,
    N_mc,
    sample_next,
    durations,
    idle_est,
    start_prev_act=None,
):
    bins = int(math.ceil(horizon_s / bin_s))
    # For each MC, produce matrix [R x bins] busy seconds
    res_list = list(resources)
    R = len(res_list)
    workload_mc = []
    for mc in range(N_mc):
        mat = np.zeros((R, bins), dtype=np.float32)
        for ri, r in enumerate(res_list):
            t = 0.0
            prev_act = start_prev_act.get(r, None) if start_prev_act else None
            # initialize prev_act as most common in durations if None
            if prev_act is None and len(durations) > 0:
                prev_act = max(durations.keys(), key=lambda k: durations[k])
            while t < horizon_s:
                current_ts = start_time + timedelta(seconds=t)
                na = sample_next(
                    str(r), str(prev_act) if prev_act else "", current_ts
                ) or (prev_act or next(iter(durations.keys())))
                dur = float(
                    durations.get(
                        na, np.median(list(durations.values())) if durations else 1800.0
                    )
                )
                # fill bins
                start = t
                end = min(t + dur, horizon_s)
                b0 = int(start // bin_s)
                b1 = int((end - 1e-6) // bin_s)
                for b in range(b0, b1 + 1):
                    bin_start = b * bin_s
                    bin_end = (b + 1) * bin_s
                    overlap = max(0.0, min(end, bin_end) - max(start, bin_start))
                    if overlap > 0:
                        mat[ri, b] += overlap
                t = end + float(
                    idle_est.get(
                        r,
                        np.median(
                            list(idle_est.values()) if len(idle_est) > 0 else [300.0]
                        ),
                    )
                )
                prev_act = na
        workload_mc.append(mat)
    workload_pred = np.mean(np.stack(workload_mc, axis=0), axis=0)  # [R,B]
    return res_list, workload_pred


def actual_future_workload(df_test, resources, start_time, horizon_s, bin_s, durations):
    df = df_test[
        (df_test["timestamp"] >= start_time)
        & (df_test["timestamp"] < start_time + timedelta(seconds=horizon_s))
    ].copy()
    df = df.sort_values("timestamp")
    bins = int(math.ceil(horizon_s / bin_s))
    res_list = list(resources)
    idx = {r: i for i, r in enumerate(res_list)}
    mat = np.zeros((len(res_list), bins), dtype=np.float32)
    for _, row in df.iterrows():
        r = str(row.get("resource", "System"))
        if r not in idx:
            continue
        a = str(row["activity"])
        dur = float(
            durations.get(
                a, np.median(list(durations.values())) if durations else 1800.0
            )
        )
        t0 = (row["timestamp"] - start_time).total_seconds()
        if t0 < 0:
            continue
        t1 = min(t0 + dur, horizon_s)
        b0 = int(t0 // bin_s)
        b1 = int((t1 - 1e-6) // bin_s)
        for b in range(b0, min(bins - 1, b1) + 1):
            bin_start = b * bin_s
            bin_end = (b + 1) * bin_s
            overlap = max(0.0, min(t1, bin_end) - max(t0, bin_start))
            if overlap > 0:
                mat[idx[r], b] += overlap
    return res_list, mat


def compute_wmape(actual, pred, eps=1e-6):
    # actual, pred: [R,B] busy seconds
    weights = actual  # weight by actual workload per resource per bin
    num = np.sum(weights * np.abs(pred - actual))
    den = np.sum(weights * (actual + eps))
    if den <= 0:
        return 1.0
    return float(num / den)


def train_one_dataset(
    name, df, max_epochs=8, batch_size=128, max_prefix_len=10, lr=1e-3
):
    print(f"\n=== Dataset: {name} ===")
    train_cases, val_cases, test_cases = time_based_split(df, 0.7, 0.15)
    samples_all, act2id, id2act, pad_id = build_prefix_dataset(
        df, max_prefix_len=max_prefix_len
    )
    samples_train = [s for s in samples_all if s["case_id"] in train_cases]
    samples_val = [s for s in samples_all if s["case_id"] in val_cases]
    samples_test = [s for s in samples_all if s["case_id"] in test_cases]
    # Re-normalize on train
    if len(samples_train) > 0:
        all_feats = np.concatenate(
            [s["seq_feats"] for s in samples_train if s["seq_feats"].shape[0] > 0],
            axis=0,
        )
        dt_mean, dt_std = all_feats[:, 0].mean(), all_feats[:, 0].std() + 1e-6
        ss_mean, ss_std = all_feats[:, 1].mean(), all_feats[:, 1].std() + 1e-6

        def norm(samples):
            for s in samples:
                if s["seq_feats"].shape[0] > 0:
                    s["seq_feats"][:, 0] = (s["seq_feats"][:, 0] - dt_mean) / dt_std
                    s["seq_feats"][:, 1] = (s["seq_feats"][:, 1] - ss_mean) / ss_std

        norm(samples_train)
        norm(samples_val)
        norm(samples_test)
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

    # Prepare resource-centric artifacts from training portion only
    df_sorted = df.sort_values("timestamp")
    df_train = df_sorted[df_sorted["case_id"].isin(train_cases)].copy()
    df_val = df_sorted[df_sorted["case_id"].isin(val_cases)].copy()
    df_test = df_sorted[df_sorted["case_id"].isin(test_cases)].copy()
    durations, global_med = estimate_activity_durations(df_train)
    sample_next, idle_est = build_resource_policy(df_train)
    resources = df_sorted["resource"].astype(str).fillna("System").unique().tolist()
    if len(resources) == 0:
        resources = ["System"]

    # Choose workload windows for wMAPE
    # Start at earliest test time; also a couple more offsets if possible
    H = 60 * 60  # 60 minutes
    bin_s = 10 * 60  # 10-minute bins
    starts = df_test["timestamp"].sort_values().dropna().unique()
    start_times = []
    if len(starts) > 0:
        start_times.append(pd.to_datetime(starts[0]))
        if len(starts) > 100:
            start_times.append(pd.to_datetime(starts[len(starts) // 3]))
            start_times.append(pd.to_datetime(starts[2 * len(starts) // 3]))
    # Precompute baseline wMAPE once (will be logged each epoch)
    wmape_epoch = 0.0
    if len(start_times) > 0:
        wmapes = []
        for st in start_times:
            res_pred_list = []
            res_actual_list = []
            res_list_pred, pred_mat = simulate_resource_workload(
                resources,
                st.to_pydatetime(),
                H,
                bin_s,
                N_mc=10,
                sample_next=sample_next,
                durations=durations,
                idle_est=idle_est,
            )
            res_list_act, act_mat = actual_future_workload(
                df_test, resources, st.to_pydatetime(), H, bin_s, durations
            )
            # align order
            pred_mat = pred_mat
            act_mat = act_mat
            w = compute_wmape(act_mat, pred_mat)
            wmapes.append(w)
            experiment_data[name]["resource_workload"]["pred"].append(pred_mat)
            experiment_data[name]["resource_workload"]["actual"].append(act_mat)
        wmape_epoch = float(np.mean(wmapes)) if len(wmapes) > 0 else 1.0
    else:
        wmape_epoch = 1.0

    # Training loop with per-epoch metrics and wMAPE logging
    best_val_top3 = -1.0
    best_state = None
    hist = {"train_loss": [], "val_loss": [], "val_top3": []}
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
        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
        print(
            f"  val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | val_top3={val_top3:.4f} | wMAPE@H={wmape_epoch:.4f}"
        )
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["val_top3"].append(val_top3)
        experiment_data[name]["losses"]["train"].append((epoch, train_loss))
        experiment_data[name]["losses"]["val"].append((epoch, val_loss))
        experiment_data[name]["metrics"]["val"].append(
            (
                epoch,
                {
                    "acc": val_acc,
                    "macro_f1": val_f1,
                    "top3": val_top3,
                    "wMAPE@H": wmape_epoch,
                },
            )
        )
        experiment_data[name]["epochs"].append(epoch)
        # track best
        if val_top3 > best_val_top3:
            best_val_top3 = val_top3
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
    experiment_data[name]["metrics"]["train"].append(
        (
            "final",
            {
                "loss": train_loss,
                "acc": train_acc,
                "macro_f1": train_f1,
                "top3": train_top3,
                "wMAPE@H": wmape_epoch,
            },
        )
    )
    experiment_data[name]["metrics"]["val"].append(
        (
            "final",
            {
                "loss": val_loss,
                "acc": val_acc,
                "macro_f1": val_f1,
                "top3": val_top3,
                "wMAPE@H": wmape_epoch,
            },
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
                "wMAPE@H": wmape_epoch,
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
    except Exception as e:
        print(f"[warn] Plotting failed: {e}")

    # Save confusion-matrix-like data
    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true_t, y_pred_t)
        np.save(os.path.join(working_dir, f"cm_{name}.npy"), cm)
    except Exception as e:
        print(f"[warn] Confusion matrix failed: {e}")


def run():
    datasets = load_datasets()
    # iterate datasets with possible cap for speed
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
                df_small = df.copy()
        except Exception:
            df_small = df.copy()
        train_one_dataset(
            key, df_small, max_epochs=8, batch_size=128, max_prefix_len=10, lr=1e-3
        )
    # Save experiment data
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    np.savez_compressed(
        os.path.join(working_dir, "experiment_data_compressed.npz"),
        data=experiment_data,
    )


# Execute immediately
run()
