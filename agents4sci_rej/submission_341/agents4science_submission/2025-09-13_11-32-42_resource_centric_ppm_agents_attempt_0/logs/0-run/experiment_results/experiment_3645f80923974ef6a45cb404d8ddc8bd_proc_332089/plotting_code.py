import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import timezone
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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


def load_local_xes():
    try:
        from ai_scientist.ideas.my_research_topic import (
            load_datasets,
            pick_default_dataset,
        )

        datasets = load_datasets()
        name, df = pick_default_dataset(datasets)
        return name, df
    except Exception as e:
        print(f"[fallback] Helper loader failed: {e}")
        try:
            from pm4py.objects.log.importer.xes import importer as xes_importer
        except Exception as e2:
            raise ImportError("pm4py is required. Install: pip install pm4py") from e2
        base = Path("input")
        if not base.exists():
            raise FileNotFoundError("No input directory with XES files found.")
        xes_files = list(base.glob("*.xes")) + list(base.glob("*.xes.gz"))
        if not xes_files:
            raise FileNotFoundError("No .xes or .xes.gz files in input directory.")
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

if "lifecycle" in df.columns:
    if df["lifecycle"].notna().any():
        if (df["lifecycle"] == "complete").any():
            df = df[df["lifecycle"] == "complete"].copy()

df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

traces = []
case_start_times = {}
for cid, g in df.groupby("case_id"):
    g = g.sort_values("timestamp")
    acts = g["activity"].tolist()
    times = pd.to_datetime(g["timestamp"]).tolist()
    if len(acts) >= 2:
        traces.append((cid, acts, times))
        case_start_times[cid] = times[0]

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

from itertools import chain

train_acts = list(chain.from_iterable([t[1] for t in train_cases]))
act_counts = pd.Series(train_acts).value_counts()
act_list = act_counts.index.tolist()
PAD = "<PAD>"
UNK = "<UNK>"
itos = [PAD, UNK] + act_list
stoi = {a: i for i, a in enumerate(itos)}
pad_idx = stoi[PAD]
unk_idx = stoi[UNK]
num_classes = len(itos)


def act_to_idx(a):
    return stoi.get(a, unk_idx)


def build_prefix_samples(cases):
    samples = []
    for cid, acts, times in cases:
        t0 = times[0]
        for t in range(1, len(acts)):
            prefix_acts = acts[:t]
            prefix_times = times[:t]
            target = acts[t]
            feats = []
            for i, ts in enumerate(prefix_times):
                delta_case = (ts - t0).total_seconds()
                delta_prev = (
                    (ts - prefix_times[i - 1]).total_seconds() if i > 0 else 0.0
                )
                dt = pd.Timestamp(ts)
                if dt.tzinfo is None:
                    dt = dt.tz_localize("UTC")
                else:
                    dt = dt.tz_convert("UTC")
                feats.append([delta_case, delta_prev, dt.hour, dt.weekday()])
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


def normalize_feats(f):
    f = f.copy()
    f[:, 0:2] = (f[:, 0:2] - num_mean[0:2]) / num_std[0:2]
    f[:, 2] = (f[:, 2] - num_mean[2]) / num_std[2]
    f[:, 3] = (f[:, 3] - num_mean[3]) / num_std[3]
    return f


class PrefixDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, act_seq, feats, target = self.samples[idx]
        feats = normalize_feats(feats)
        return {
            "act_seq": torch.tensor(act_seq, dtype=torch.long),
            "feats": torch.tensor(feats, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.long),
            "length": torch.tensor(len(act_seq), dtype=torch.long),
        }


def collate_fn(batch):
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


class LSTMNextActivity(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_idx,
        emb_dim=64,
        hidden_dim=128,
        num_feat=4,
        num_classes=0,
        dropout=0.2,
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
        emb = self.emb(act_seq)
        x = torch.cat([emb, feats], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = self.dropout(h_n[-1])
        return self.fc(h_last)


model = LSTMNextActivity(num_classes, pad_idx).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


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
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_targets.append(batch["target"].cpu().numpy())
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_true = np.concatenate(all_targets) if all_targets else np.array([])
    avg_loss = total_loss / max(1, len(y_true))
    acc = accuracy_score(y_true, y_pred) if y_true.size > 0 else 0.0
    macro_f1 = (
        f1_score(y_true, y_pred, average="macro", zero_division=0)
        if y_true.size > 0
        else 0.0
    )
    # top-3
    top3 = 0.0
    if y_true.size > 0:
        t3 = 0
        n = 0
        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                logits = model(batch["act_seq"], batch["feats"], batch["lengths"])
                topk = torch.topk(logits, k=min(3, logits.size(1)), dim=1).indices
                t3 += (topk == batch["target"].unsqueeze(1)).any(1).sum().item()
                n += batch["target"].size(0)
        top3 = t3 / max(1, n)
    return avg_loss, acc, macro_f1, top3, y_pred, y_true


epochs = 8
best_val_loss = float("inf")
for epoch in range(1, epochs + 1):
    model.train()
    running = 0.0
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
        running += loss.item() * batch["target"].size(0)
    train_loss = running / max(1, len(train_ds))
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
        torch.save(
            model.state_dict(),
            os.path.join(working_dir, f"best_model_{dataset_name}.pt"),
        )

best_path = os.path.join(working_dir, f"best_model_{dataset_name}.pt")
if os.path.exists(best_path):
    model.load_state_dict(
        torch.load(best_path, map_mode=device, map_location=device)
        if hasattr(torch.load, "__call__")
        else torch.load(best_path, map_location=device)
    )

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
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print(f"Top-3 Next-Activity Accuracy (test) = {test_top3:.4f}")

# ---------- Plotting from experiment_data.npy only ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    ds_key = "default_dataset"
    meta = experiment_data.get(ds_key, {}).get("meta", {})
    dname = meta.get("dataset_name", "dataset")
    # 1) Loss curves
    try:
        plt.figure()
        losses = experiment_data[ds_key]["losses"]
        if losses["train"]:
            ep, tr = zip(*losses["train"])
            plt.plot(ep, tr, label="Train Loss")
        if losses["val"]:
            epv, vl = zip(*losses["val"])
            plt.plot(epv, vl, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname} - Training/Validation Loss\nNext-Activity Prediction (BPM)")
        plt.legend()
        out = os.path.join(working_dir, f"{dname}_loss_curves_next_activity.png")
        plt.savefig(out)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # 2) Metric curves (Acc, Macro-F1, Top-3)
    try:
        plt.figure()
        mtr = experiment_data[ds_key]["metrics"]["train"]
        mval = experiment_data[ds_key]["metrics"]["val"]
        if mtr:
            ep_t, acc_t, f1_t, top3_t = zip(*mtr)
            plt.plot(ep_t, acc_t, label="Train Acc")
            plt.plot(ep_t, f1_t, label="Train Macro-F1")
            plt.plot(ep_t, top3_t, label="Train Top-3")
        if mval:
            ep_v, acc_v, f1_v, top3_v = zip(*mval)
            plt.plot(ep_v, acc_v, "--", label="Val Acc")
            plt.plot(ep_v, f1_v, "--", label="Val Macro-F1")
            plt.plot(ep_v, top3_v, "--", label="Val Top-3")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(
            f"{dname} - Training/Validation Metrics\nNext-Activity Prediction (BPM)"
        )
        plt.legend()
        out = os.path.join(working_dir, f"{dname}_metric_curves_next_activity.png")
        plt.savefig(out)
        plt.close()
    except Exception as e:
        print(f"Error creating metric curves: {e}")
        plt.close()

    # 3) Confusion Matrix (Test)
    try:
        y_true = experiment_data[ds_key].get("ground_truth", [])
        y_pred = experiment_data[ds_key].get("predictions", [])
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_true.size > 0 and y_pred.size > 0:
            cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.colorbar(im)
            plt.title(
                f"{dname} - Confusion Matrix (Test)\nNext-Activity Prediction (BPM)"
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            ticks = np.arange(len(np.unique(y_true)))
            plt.xticks(ticks, ticks, rotation=90)
            plt.yticks(ticks, ticks)
            plt.tight_layout()
            out = os.path.join(
                working_dir, f"{dname}_confusion_matrix_next_activity.png"
            )
            plt.savefig(out)
            plt.close()
        else:
            print("Confusion Matrix skipped: no predictions/ground truth available.")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
