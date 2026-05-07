import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from datetime import datetime
import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experiment data container
experiment_data = {
    "BPI_base": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "timestamp": [],
    }
}

# ------------- Data discovery and loading (pm4py XES) -------------
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _has_xes(dirpath: Path) -> bool:
    try:
        return dirpath.is_dir() and (
            any(dirpath.glob("*.xes")) or any(dirpath.glob("*.xes.gz"))
        )
    except Exception:
        return False


def _resolve_data_dir() -> Path:
    candidates: List[Path] = []
    candidates += [Path("input").resolve(), (Path.cwd() / "input").resolve()]
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        candidates.append((base / "data").resolve())
        candidates.append((base / "input").resolve())
    candidates += [
        Path("/workspace/input"),
        Path("/workspace/data"),
        Path("/workspace/ai_scientist/data"),
        Path("/workspace/AI-Scientist-v2/data"),
        Path("/workspace/experiments/data"),
        Path("/workspace/ai_scientist/input"),
        Path("/workspace/experiments/input"),
    ]
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if _has_xes(p):
            print(f"[data] Using discovered data dir: {p}")
            return p
    raise FileNotFoundError("No directory containing .xes or .xes.gz found.")


def _first_match(d: Path, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        for p in d.glob(pat):
            if p.is_file():
                return p
    return None


def xes_to_df(xes_path: Path) -> pd.DataFrame:
    try:
        from pm4py.objects.log.importer.xes import importer as xes_importer
    except Exception as e:
        raise ImportError("pm4py is required. Install with `pip install pm4py`.") from e
    print(f"[data] Loading XES: {xes_path}")
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
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    return df[["case_id", "activity", "lifecycle", "timestamp", "resource"]]


def load_default_dataset() -> Tuple[str, pd.DataFrame]:
    try:
        d = _resolve_data_dir()
        avail = sorted(
            [p.name for p in list(d.glob("*.xes")) + list(d.glob("*.xes.gz"))]
        )
        print(f"[data] Available: {avail}")
        patterns = {
            "BPI2017": ["BPI_Challenge_2017*.xes*", "BPI2017*.xes*", "*2017*.xes*"],
            "BPI2012": ["BPI_Challenge_2012*.xes*", "BPI2012*.xes*", "*2012*.xes*"],
            "ROAD": [
                "Road_Traffic_Fine_Management_Process*.xes*",
                "*Traffic*Fine*.xes*",
                "*Traffic*.xes*",
            ],
        }
        for name in ("BPI2012", "BPI2017", "ROAD"):
            path = _first_match(d, patterns[name])
            if path is not None:
                return name, xes_to_df(path)
        # fallback any file
        files = list(d.glob("*.xes")) + list(d.glob("*.xes.gz"))
        if files:
            return files[0].stem, xes_to_df(files[0])
        raise FileNotFoundError("No XES files matched.")
    except Exception as e:
        print(f"[warn] XES load failed: {e}. Generating synthetic toy log.")
        # Synthetic tiny log: 3 activities, 50 cases
        np.random.seed(42)
        acts = ["A", "B", "C", "D", "E"]
        rows = []
        start = pd.Timestamp("2020-01-01", tz="UTC")
        for cid in range(50):
            length = np.random.randint(3, 7)
            t = (
                start
                + pd.Timedelta(days=np.random.randint(0, 30))
                + pd.Timedelta(minutes=np.random.randint(0, 1440))
            )
            case_id = f"C{cid:04d}"
            seq = ["A"] + list(np.random.choice(acts[1:], size=length - 1))
            for a in seq:
                rows.append(
                    {
                        "case_id": case_id,
                        "activity": a,
                        "lifecycle": "complete",
                        "timestamp": t,
                        "resource": np.random.choice(["R1", "R2", "R3"]),
                    }
                )
                t = t + pd.Timedelta(minutes=np.random.randint(1, 120))
        df = (
            pd.DataFrame(rows)
            .sort_values(["case_id", "timestamp"])
            .reset_index(drop=True)
        )
        return "SYNTHETIC", df


# ------------- Prefix building and features -------------
def build_prefix_samples(df: pd.DataFrame, max_k: int = 10):
    # Ensure per-case time order
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    # Case start times for time-based split
    case_start = (
        df.groupby("case_id")["timestamp"]
        .min()
        .reset_index()
        .rename(columns={"timestamp": "case_start"})
    )
    # Build sequences per case
    samples = []
    for cid, grp in df.groupby("case_id"):
        grp = grp.sort_values("timestamp")
        acts = grp["activity"].tolist()
        times = grp["timestamp"].tolist()
        res = grp["resource"].tolist()
        if len(acts) < 2:
            continue
        for k in range(1, min(len(acts), max_k + 1)):
            prefix_acts = acts[:k]
            prefix_times = times[:k]
            target = acts[k] if k < len(acts) else None
            if target is None:
                continue
            # per-step time features
            t0 = prefix_times[0]
            times_since_start = [
                (t - t0).total_seconds() / 3600.0 for t in prefix_times
            ]
            times_since_prev = [0.0] + [
                (prefix_times[i] - prefix_times[i - 1]).total_seconds() / 3600.0
                for i in range(1, len(prefix_times))
            ]
            hours = [t.hour + t.minute / 60.0 for t in prefix_times]
            weekdays = [t.weekday() for t in prefix_times]
            working = [
                1.0 if (wd < 5 and 8 <= h <= 18) else 0.0
                for wd, h in zip(weekdays, hours)
            ]
            samples.append(
                {
                    "case_id": cid,
                    "prefix_acts": prefix_acts,
                    "times_since_start": times_since_start,
                    "times_since_prev": times_since_prev,
                    "hours": hours,
                    "weekdays": weekdays,
                    "working": working,
                    "target": target,
                    "case_start": t0,
                }
            )
    return pd.DataFrame(samples), case_start


# ------------- Vocabulary and encoding -------------
class Vocab:
    def __init__(self, tokens, add_unk=True, add_pad=True):
        uniq = sorted(set(tokens))
        self.pad_token = "<PAD>" if add_pad else None
        self.unk_token = "<UNK>" if add_unk else None
        idx = 0
        self.token_to_id = {}
        if add_pad:
            self.token_to_id[self.pad_token] = idx
            idx += 1
        if add_unk:
            self.token_to_id[self.unk_token] = idx
            idx += 1
        for t in uniq:
            if t in (self.pad_token, self.unk_token):
                continue
            self.token_to_id[t] = idx
            idx += 1
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def __len__(self):
        return len(self.token_to_id)

    def encode(self, t):
        if t in self.token_to_id:
            return self.token_to_id[t]
        if self.unk_token is not None:
            return self.token_to_id[self.unk_token]
        raise KeyError(t)

    def pad_id(self):
        return self.token_to_id[self.pad_token] if self.pad_token else 0


# ------------- Dataset and collate -------------
class PrefixDataset(Dataset):
    def __init__(self, df, act_vocab: Vocab, target_vocab: Vocab, norm_stats: dict):
        self.df = df
        self.act_vocab = act_vocab
        self.target_vocab = target_vocab
        self.norm = norm_stats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        acts = torch.tensor(
            [self.act_vocab.encode(a) for a in row["prefix_acts"]], dtype=torch.long
        )
        # numeric features per step
        feats = np.stack(
            [
                row["times_since_start"],
                row["times_since_prev"],
                row["hours"],
                row["weekdays"],
                row["working"],
            ],
            axis=1,
        ).astype(np.float32)
        # normalize using train stats
        mu = self.norm["mean"]  # shape (5,)
        sigma = self.norm["std"]
        feats = (feats - mu) / (sigma + 1e-8)
        feats = torch.tensor(feats, dtype=torch.float32)
        y = torch.tensor(self.target_vocab.encode(row["target"]), dtype=torch.long)
        return {"acts": acts, "feats": feats, "y": y}


def collate_batch(batch, pad_id, feat_dim):
    lengths = [len(b["acts"]) for b in batch]
    max_len = max(lengths)
    B = len(batch)
    acts = torch.full((B, max_len), pad_id, dtype=torch.long)
    feats = torch.zeros((B, max_len, feat_dim), dtype=torch.float32)
    ys = torch.zeros((B,), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["acts"])
        acts[i, :L] = b["acts"]
        feats[i, :L] = b["feats"]
        ys[i] = b["y"]
    return {
        "acts": acts,
        "feats": feats,
        "y": ys,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


# ------------- Model -------------
class LSTMNextAct(nn.Module):
    def __init__(self, num_acts, emb_dim, feat_dim, hidden, num_classes, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(num_acts, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim + feat_dim, hidden_size=hidden, batch_first=True
        )
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, acts, feats, lengths):
        x_emb = self.emb(acts)  # (B,T,E)
        x = torch.cat([x_emb, feats], dim=-1)
        # pack for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, (h, c) = self.lstm(packed)
        h_last = h[-1]  # (B,H)
        h_last = self.drop(h_last)
        logits = self.fc(h_last)
        return logits


# ------------- Metrics -------------
def topk_accuracy(probs, y_true, k=3):
    topk = probs.topk(k, dim=1).indices  # (B,k)
    correct = (topk == y_true.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


def eval_model(model, loader, criterion):
    model.eval()
    all_logits = []
    all_y = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["acts"], batch["feats"], batch["lengths"])
            loss = criterion(logits, batch["y"])
            losses.append(loss.item())
            all_logits.append(logits.cpu())
            all_y.append(batch["y"].cpu())
    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    probs = torch.softmax(logits, dim=1)
    y_pred = probs.argmax(dim=1).numpy()
    y_true = y.numpy()
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    top3 = topk_accuracy(probs, y, k=3)
    return np.mean(losses), acc, macro_f1, top3, probs.numpy(), y_true


# ------------- Pipeline -------------
def run_experiment():
    dataset_name, df = load_default_dataset()
    print(
        f"[info] Using dataset: {dataset_name}, events={len(df)}, cases={df['case_id'].nunique()}"
    )
    samples_df, case_start = build_prefix_samples(df, max_k=10)
    if len(samples_df) == 0:
        raise RuntimeError("No samples could be built from the event log.")
    # Time-based split by case start
    case_times = samples_df.groupby("case_id")["case_start"].min().reset_index()
    case_times = case_times.sort_values("case_start")
    n_cases = len(case_times)
    n_train = int(0.7 * n_cases)
    n_val = int(0.15 * n_cases)
    train_cases = set(case_times.iloc[:n_train]["case_id"])
    val_cases = set(case_times.iloc[n_train : n_train + n_val]["case_id"])
    test_cases = set(case_times.iloc[n_train + n_val :]["case_id"])
    train_df = samples_df[samples_df["case_id"].isin(train_cases)].reset_index(
        drop=True
    )
    val_df = samples_df[samples_df["case_id"].isin(val_cases)].reset_index(drop=True)
    test_df = samples_df[samples_df["case_id"].isin(test_cases)].reset_index(drop=True)
    print(
        f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)} samples"
    )

    # Vocabularies (fit on TRAIN ONLY for targets; activity embeddings can include all seen to avoid UNK explosion)
    act_vocab = Vocab(tokens=df["activity"].tolist(), add_unk=True, add_pad=True)
    target_vocab = Vocab(
        tokens=train_df["target"].tolist(), add_unk=True, add_pad=False
    )

    # Compute normalization stats on TRAIN numeric features (stack all steps)
    def stack_numeric(df_):
        arrs = []
        for _, r in df_.iterrows():
            feats = np.stack(
                [
                    r["times_since_start"],
                    r["times_since_prev"],
                    r["hours"],
                    r["weekdays"],
                    r["working"],
                ],
                axis=1,
            ).astype(np.float32)
            arrs.append(feats)
        if len(arrs) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return np.concatenate(arrs, axis=0)

    train_feats_all = stack_numeric(train_df)
    if train_feats_all.shape[0] == 0:
        raise RuntimeError("Training features are empty.")
    mu = train_feats_all.mean(axis=0)
    std = train_feats_all.std(axis=0)
    std[std == 0] = 1.0
    norm_stats = {"mean": mu, "std": std}

    # Datasets and loaders
    feat_dim = 5
    pad_id = act_vocab.pad_id()
    train_ds = PrefixDataset(train_df, act_vocab, target_vocab, norm_stats)
    val_ds = PrefixDataset(val_df, act_vocab, target_vocab, norm_stats)
    test_ds = PrefixDataset(test_df, act_vocab, target_vocab, norm_stats)
    collate = lambda b: collate_batch(b, pad_id=pad_id, feat_dim=feat_dim)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate)

    # Model
    num_acts = len(act_vocab)
    num_classes = len(target_vocab)
    model = LSTMNextAct(
        num_acts=num_acts,
        emb_dim=64,
        feat_dim=feat_dim,
        hidden=64,
        num_classes=num_classes,
        pad_idx=pad_id,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    epochs = 8
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["acts"], batch["feats"], batch["lengths"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        # Validation
        val_loss, val_acc, val_f1, val_top3, _, _ = eval_model(
            model, val_loader, criterion
        )
        print(
            f"Epoch {epoch}: train_loss = {train_loss:.4f} | validation_loss = {val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | val_top3={val_top3:.4f}"
        )
        # Record metrics
        ts = time.time()
        experiment_data["BPI_base"]["epochs"].append(epoch)
        experiment_data["BPI_base"]["timestamp"].append(ts)
        experiment_data["BPI_base"]["losses"]["train"].append(
            {"epoch": epoch, "loss": train_loss, "t": ts}
        )
        experiment_data["BPI_base"]["losses"]["val"].append(
            {"epoch": epoch, "loss": val_loss, "t": ts}
        )
        experiment_data["BPI_base"]["metrics"]["train"].append(
            {"epoch": epoch, "metric": "loss", "value": train_loss, "t": ts}
        )
        experiment_data["BPI_base"]["metrics"]["val"].append(
            {
                "epoch": epoch,
                "accuracy": val_acc,
                "macro_f1": val_f1,
                "top3_accuracy": val_top3,
                "validation_loss": val_loss,
                "t": ts,
            }
        )

    # Final test evaluation
    test_loss, test_acc, test_f1, test_top3, test_probs, test_y = eval_model(
        model, test_loader, criterion
    )
    print(
        f"[test] loss={test_loss:.4f} acc={test_acc:.4f} macro_f1={test_f1:.4f} top3_acc={test_top3:.4f}"
    )

    # Save predictions and ground truth (use class indices)
    experiment_data["BPI_base"]["predictions"] = test_probs
    experiment_data["BPI_base"]["ground_truth"] = test_y.tolist()
    experiment_data["BPI_base"]["metrics"]["test"].append(
        {
            "loss": test_loss,
            "accuracy": test_acc,
            "macro_f1": test_f1,
            "top3_accuracy": test_top3,
            "t": time.time(),
            "dataset": dataset_name,
        }
    )

    # Save all experiment data
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    # Also save compressed to be safe
    np.savez_compressed(
        os.path.join(working_dir, f"next_act_results_{dataset_name}.npz"),
        experiment_data=experiment_data,
    )


# Execute
run_experiment()
