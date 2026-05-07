import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import timezone
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore")

# Device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experiment data structure
experiment_data = {
    "default_dataset": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "meta": {},
    }
}


# ------------- Data Loading -------------
def load_local_xes():
    try:
        # Prefer helper if available
        from ai_scientist.ideas.my_research_topic import (
            load_datasets,
            pick_default_dataset,
        )

        datasets = load_datasets()
        name, df = pick_default_dataset(datasets)
        return name, df
    except Exception as e:
        print(f"[fallback] Helper loader failed: {e}")
        # Fallback: search input dir
        try:
            from pm4py.objects.log.importer.xes import importer as xes_importer
        except Exception as e2:
            raise ImportError("pm4py is required. Install: pip install pm4py") from e2
        # find any xes
        base = Path("input")
        if not base.exists():
            raise FileNotFoundError("No input directory with XES files found.")
        xes_files = list(base.glob("*.xes")) + list(base.glob("*.xes.gz"))
        if not xes_files:
            raise FileNotFoundError("No .xes or .xes.gz files in input directory.")
        # pick first
        xes_path = xes_files[0]
        print(f"[fallback] Loading XES: {xes_path}")
        log = xes_importer.apply(str(xes_path))
        rows = []
        for tr in log:
            case_id = tr.attributes.get("concept:name") or tr.attributes.get(
                "case:concept:name"
            )
            for e in tr:
                rows.append(
                    {
                        "case_id": case_id,
                        "activity": e.get("concept:name"),
                        "lifecycle": e.get("lifecycle:transition", "complete"),
                        "timestamp": e.get("time:timestamp"),
                        "resource": e.get("org:resource", "System"),
                    }
                )
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
        df = df[["case_id", "activity", "lifecycle", "timestamp", "resource"]]
        df = df.sort_values(["timestamp", "case_id"]).reset_index(drop=True)
        return xes_path.stem, df


dataset_name, df = load_local_xes()
print(f"[data] Using dataset: {dataset_name}, shape={df.shape}")

# ------------- Preprocessing & Prefix Generation -------------
# Keep only 'complete' events if lifecycle exists
if "lifecycle" in df.columns:
    if df["lifecycle"].notna().any():
        # Many logs use 'complete'; if missing keep all
        if (df["lifecycle"] == "complete").any():
            df = df[df["lifecycle"] == "complete"].copy()

# Sort within case by timestamp
df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

# Build traces and case start times
traces = []
case_start_times = {}
for cid, g in df.groupby("case_id"):
    g = g.sort_values("timestamp")
    acts = g["activity"].tolist()
    times = pd.to_datetime(g["timestamp"]).tolist()
    if len(acts) >= 2:
        traces.append((cid, acts, times))
        case_start_times[cid] = times[0]

# Time-based split at case level
cases_sorted = sorted(traces, key=lambda x: case_start_times[x[0]])
n = len(cases_sorted)
if n == 0:
    raise RuntimeError("No usable traces with length >= 2 found.")
train_end = int(0.7 * n)
val_end = int(0.85 * n)
train_cases = cases_sorted[:train_end]
val_cases = cases_sorted[train_end:val_end]
test_cases = cases_sorted[val_end:]

print(
    f"[split] Cases: train={len(train_cases)}, val={len(val_cases)}, test={len(test_cases)}"
)

# Build activity vocab from training only to avoid leakage
from itertools import chain

train_acts = list(chain.from_iterable([t[1] for t in train_cases]))
act_counts = pd.Series(train_acts).value_counts()
act_list = act_counts.index.tolist()
# Add UNK for unseen in val/test
PAD = "<PAD>"
UNK = "<UNK>"
itos = [PAD, UNK] + act_list
stoi = {a: i for i, a in enumerate(itos)}
pad_idx = stoi[PAD]
unk_idx = stoi[UNK]
num_classes = len(itos)


def act_to_idx(a):
    return stoi.get(a, unk_idx)


# Feature normalization helpers computed on train
def build_prefix_samples(cases):
    samples = []
    for cid, acts, times in cases:
        # per-case time features relative to case start
        t0 = times[0]
        prev_time = None
        # iterate prefixes
        for t in range(1, len(acts)):  # prefix length t, target at t
            prefix_acts = acts[:t]
            prefix_times = times[:t]
            target = acts[t]
            # build per-event time features aligned with prefix positions
            feats = []
            for i, ts in enumerate(prefix_times):
                delta_case = (ts - t0).total_seconds()
                delta_prev = (
                    (ts - prefix_times[i - 1]).total_seconds() if i > 0 else 0.0
                )
                dt = (
                    pd.Timestamp(ts).tz_convert("UTC")
                    if pd.Timestamp(ts).tzinfo
                    else pd.Timestamp(ts).tz_localize("UTC")
                )
                hour = dt.hour
                weekday = dt.weekday()
                feats.append([delta_case, delta_prev, hour, weekday])
            samples.append(
                (
                    cid,
                    [act_to_idx(a) for a in prefix_acts],
                    np.array(feats, dtype=np.float32),
                    act_to_idx(target),
                )
            )
    return samples


train_samples = build_prefix_samples(train_cases)
val_samples = build_prefix_samples(val_cases)
test_samples = build_prefix_samples(test_cases)

print(
    f"[samples] train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
)


# Fit normalizers on train for numerical features: delta_case, delta_prev, hour, weekday
def compute_norm_stats(samples):
    X = (
        np.concatenate([s[2] for s in samples], axis=0)
        if samples
        else np.zeros((0, 4), dtype=np.float32)
    )
    if X.shape[0] == 0:
        return np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


num_mean, num_std = compute_norm_stats(train_samples)


def normalize_feats(feats):
    # feats: [T,4]
    f = feats.copy()
    # Normalize seconds-based first two; scale hour and weekday to 0-1
    f[:, 0:2] = (f[:, 0:2] - num_mean[0:2]) / num_std[0:2]
    f[:, 2] = (f[:, 2] - num_mean[2]) / (num_std[2])
    f[:, 3] = (f[:, 3] - num_mean[3]) / (num_std[3])
    return f


# ------------- Dataset & DataLoader -------------
class PrefixDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cid, act_seq, feats, target = self.samples[idx]
        feats = normalize_feats(feats)
        return {
            "act_seq": torch.tensor(act_seq, dtype=torch.long),
            "feats": torch.tensor(feats, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.long),
            "length": torch.tensor(len(act_seq), dtype=torch.long),
        }


def collate_fn(batch):
    # pad sequences
    lengths = torch.tensor([b["length"].item() for b in batch], dtype=torch.long)
    max_len = lengths.max().item()
    B = len(batch)
    act_pad = torch.full((B, max_len), pad_idx, dtype=torch.long)
    feats_pad = torch.zeros((B, max_len, 4), dtype=torch.float32)
    targets = torch.tensor([b["target"].item() for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        l = b["length"].item()
        act_pad[i, :l] = b["act_seq"]
        feats_pad[i, :l, :] = b["feats"]
    return {
        "act_seq": act_pad,
        "feats": feats_pad,
        "lengths": lengths,
        "target": targets,
    }


# For speed, optionally cap number of samples
max_train_samples = 60000
max_val_samples = 15000
max_test_samples = 20000
if len(train_samples) > max_train_samples:
    train_samples = train_samples[:max_train_samples]
if len(val_samples) > max_val_samples:
    val_samples = val_samples[:max_val_samples]
if len(test_samples) > max_test_samples:
    test_samples = test_samples[:max_test_samples]

train_ds = PrefixDataset(train_samples)
val_ds = PrefixDataset(val_samples)
test_ds = PrefixDataset(test_samples)

batch_size = 128
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)


# ------------- Model -------------
class LSTMNextActivity(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_idx,
        emb_dim=64,
        hidden_dim=128,
        num_feat=4,
        num_classes=0,
        dropout=0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim + num_feat,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, act_seq, feats, lengths):
        # act_seq: [B,T], feats: [B,T,4], lengths: [B]
        emb = self.emb(act_seq)  # [B,T,E]
        x = torch.cat([emb, feats], dim=-1)  # [B,T,E+4]
        # pack
        lengths_cpu = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.lstm(packed)
        # Use last hidden state from LSTM
        h_last = h_n[-1]  # [B,H]
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)  # [B,C]
        return logits


model = LSTMNextActivity(
    vocab_size=num_classes,
    pad_idx=pad_idx,
    emb_dim=64,
    hidden_dim=128,
    num_feat=4,
    num_classes=num_classes,
    dropout=0.2,
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


# ------------- Metrics -------------
def topk_accuracy(logits, targets, k=3):
    with torch.no_grad():
        topk = torch.topk(logits, k=k, dim=1).indices
        correct = (topk == targets.unsqueeze(1)).any(dim=1).float()
        return correct.mean().item()


def evaluate(loader):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["act_seq"], batch["feats"], batch["lengths"])
            loss = criterion(logits, batch["target"])
            total_loss += loss.item() * batch["target"].size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(batch["target"].cpu().numpy())
    y_pred = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.array([])
    avg_loss = total_loss / max(1, len(y_true))
    acc = accuracy_score(y_true, y_pred) if y_true.size > 0 else 0.0
    macro_f1 = (
        f1_score(y_true, y_pred, average="macro", zero_division=0)
        if y_true.size > 0
        else 0.0
    )
    # compute top-3 with a second pass for exactness
    top3_acc = 0.0
    if y_true.size > 0:
        # recompute logits in mini-batches to get top-3
        t3_correct = 0
        n_total = 0
        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["act_seq"], batch["feats"], batch["lengths"])
                topk = torch.topk(logits, k=min(3, logits.size(1)), dim=1).indices
                correct = (topk == batch["target"].unsqueeze(1)).any(dim=1).sum().item()
                t3_correct += correct
                n_total += batch["target"].size(0)
        top3_acc = t3_correct / max(1, n_total)
    return avg_loss, acc, macro_f1, top3_acc, y_pred, y_true


# ------------- Training Loop -------------
epochs = 8
best_val_loss = float("inf")
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        logits = model(batch["act_seq"], batch["feats"], batch["lengths"])
        loss = criterion(logits, batch["target"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch["target"].size(0)
    train_loss = running_loss / max(1, len(train_ds))

    # Evaluate train/val
    tr_loss_eval, tr_acc, tr_f1, tr_top3, _, _ = evaluate(train_loader)
    val_loss, val_acc, val_f1, val_top3, _, _ = evaluate(val_loader)

    print(
        f"Epoch {epoch}: train_loss = {train_loss:.4f} | train_acc = {tr_acc:.4f} | train_f1 = {tr_f1:.4f} | train_top3 = {tr_top3:.4f}"
    )
    print(
        f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_acc = {val_acc:.4f} | val_f1 = {val_f1:.4f} | val_top3 = {val_top3:.4f}"
    )

    experiment_data["default_dataset"]["losses"]["train"].append(
        (epoch, float(train_loss))
    )
    experiment_data["default_dataset"]["losses"]["val"].append((epoch, float(val_loss)))
    experiment_data["default_dataset"]["metrics"]["train"].append(
        (epoch, float(tr_acc), float(tr_f1), float(tr_top3))
    )
    experiment_data["default_dataset"]["metrics"]["val"].append(
        (epoch, float(val_acc), float(val_f1), float(val_top3))
    )
    experiment_data["default_dataset"]["epochs"].append(epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save best model weights
        torch.save(
            model.state_dict(),
            os.path.join(working_dir, f"best_model_{dataset_name}.pt"),
        )

# Load best and evaluate on test
best_path = os.path.join(working_dir, f"best_model_{dataset_name}.pt")
if os.path.exists(best_path):
    model.load_state_dict(torch.load(best_path, map_location=device))

test_loss, test_acc, test_f1, test_top3, y_pred_test, y_true_test = evaluate(
    test_loader
)
print(
    f"[final] Test: loss={test_loss:.4f} | acc={test_acc:.4f} | macro_f1={test_f1:.4f} | top3_acc={test_top3:.4f}"
)

experiment_data["default_dataset"]["metrics"]["test"].append(
    ("final", float(test_acc), float(test_f1), float(test_top3))
)
experiment_data["default_dataset"]["predictions"] = y_pred_test
experiment_data["default_dataset"]["ground_truth"] = y_true_test
experiment_data["default_dataset"]["meta"] = {
    "dataset_name": dataset_name,
    "num_cases": len(cases_sorted),
    "num_train_cases": len(train_cases),
    "num_val_cases": len(val_cases),
    "num_test_cases": len(test_cases),
    "num_classes": int(num_classes),
    "vocab_size": int(num_classes),
    "primary_metric": "Top-3 Next-Activity Accuracy",
    "test_top3": float(test_top3),
    "test_accuracy": float(test_acc),
    "test_macro_f1": float(test_f1),
}

# Save experiment data
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# Also print primary metric explicitly
print(f"Top-3 Next-Activity Accuracy (test) = {test_top3:.4f}")
