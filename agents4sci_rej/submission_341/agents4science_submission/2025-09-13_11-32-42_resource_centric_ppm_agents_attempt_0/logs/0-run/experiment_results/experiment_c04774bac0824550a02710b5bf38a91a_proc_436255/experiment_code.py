import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import pandas as pd
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Device handling (CRITICAL GPU REQUIREMENTS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experiment data container
experiment_data = {}


# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ----------------- Robust XES discovery & loading -----------------
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
    tried = "\n  - " + "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate a directory containing .xes files.\n"
        f"Checked:{tried}\n"
        "Tips:\n"
        "  • Ensure filenames include BPI 2012/2017 or 'Road_Traffic_Fine_Management_Process' for auto-match."
    )


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
                    "case_id": str(case_id),
                    "activity": str(e.get("concept:name")),
                    "lifecycle": str(e.get("lifecycle:transition", "complete")),
                    "timestamp": e.get("time:timestamp"),
                    "resource": str(e.get("org:resource", "System")),
                }
            )
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    df = df[["case_id", "activity", "timestamp", "lifecycle", "resource"]]
    df = df.sort_values(["timestamp", "case_id"]).reset_index(drop=True)
    return df


def load_datasets() -> Dict[str, pd.DataFrame]:
    data_dir = _resolve_data_dir()
    available = sorted(
        [p.name for p in list(data_dir.glob("*.xes")) + list(data_dir.glob("*.xes.gz"))]
    )
    print(f"[data] Available in {data_dir}: {available}")
    patterns = {
        "BPI2012": ["BPI_Challenge_2012*.xes*", "BPI2012*.xes*", "*2012*.xes*"],
        "BPI2017": ["BPI_Challenge_2017*.xes*", "BPI2017*.xes*", "*2017*.xes*"],
        "ROAD": [
            "Road_Traffic_Fine_Management_Process*.xes*",
            "*Traffic*Fine*.xes*",
            "*Traffic*.xes*",
        ],
    }
    loaded: Dict[str, pd.DataFrame] = {}
    for key, pats in patterns.items():
        path = _first_match(data_dir, pats)
        if path is not None:
            try:
                loaded[key] = xes_to_df(path)
            except Exception as e:
                print(f"[warn] Failed to load {key} from {path}: {e}")
        else:
            print(f"[data] Not found for {key} (patterns {pats})")
    if not loaded:
        raise FileNotFoundError(
            f"No known XES files found in {data_dir}. Found: {available}"
        )
    print(f"[data] Loaded datasets: {list(loaded.keys())}")
    return loaded


# ----------------- PPM utilities -----------------
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


def build_vocab_from_df(df, use_unk=False, rare_threshold=5, train_mask=None):
    pad_id = 0
    if not use_unk:
        acts = df["activity"].astype(str).unique().tolist()
        act2id = {a: i + 1 for i, a in enumerate(sorted(acts))}
        id2act = {i: a for a, i in act2id.items()}
        unk_id = None
        return act2id, id2act, pad_id, unk_id
    else:
        if train_mask is None:
            train_mask = np.ones(len(df), dtype=bool)
        df_train = df.loc[train_mask]
        counts = df_train["activity"].astype(str).value_counts()
        frequent = set(counts[counts >= rare_threshold].index.tolist())
        sorted_freq = sorted(frequent)
        act2id = {a: i + 1 for i, a in enumerate(sorted_freq)}
        unk_id = len(act2id) + 1
        id2act = {i: a for a, i in act2id.items()}
        id2act[unk_id] = "UNK"
        return act2id, id2act, pad_id, unk_id


def build_prefix_dataset_with_vocab(
    df, act2id, unk_id=None, pad_id=0, max_prefix_len=10, min_prefix_len=1
):
    df = df.copy()
    if "lifecycle" in df.columns:
        mask = df["lifecycle"].astype(str).str.lower().eq("complete")
        if mask.any():
            df = df[mask]
    df = df.sort_values(["case_id", "timestamp"])
    samples = []
    for cid, g in df.groupby("case_id"):
        g = g.sort_values("timestamp")
        ts = (
            pd.to_datetime(g["timestamp"], utc=True).astype("int64").to_numpy() // 10**9
        )
        acts_list = g["activity"].astype(str).tolist()
        acts_ids = []
        for a in acts_list:
            if a in act2id:
                acts_ids.append(act2id[a])
            else:
                if unk_id is not None:
                    acts_ids.append(unk_id)
                else:
                    continue
        acts_ids = np.array(acts_ids, dtype=np.int64)
        if len(acts_ids) < 2:
            continue
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
        if T < 2:
            continue
        max_k = min(max_prefix_len, T - 1)
        for k in range(min_prefix_len, max_k + 1):
            seq_acts = acts_ids[:k].tolist()
            seq_feats = feats[:k]
            target = int(acts_ids[k])
            samples.append(
                {
                    "case_id": cid,
                    "seq_acts": seq_acts,
                    "seq_feats": seq_feats.copy(),
                    "target": target,
                }
            )
    if len(samples) == 0:
        return samples
    all_feats = np.concatenate(
        [s["seq_feats"] for s in samples if len(s["seq_feats"]) > 0], axis=0
    )
    dt_mean, dt_std = all_feats[:, 0].mean(), all_feats[:, 0].std() + 1e-6
    ss_mean, ss_std = all_feats[:, 1].mean(), all_feats[:, 1].std() + 1e-6
    for s in samples:
        if s["seq_feats"].shape[0] > 0:
            s["seq_feats"][:, 0] = (s["seq_feats"][:, 0] - dt_mean) / dt_std
            s["seq_feats"][:, 1] = (s["seq_feats"][:, 1] - ss_mean) / ss_std
    return samples


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


def train_one_dataset(
    name,
    df,
    ablation_type="baseline",
    unk_threshold=5,
    max_epochs=6,
    batch_size=128,
    max_prefix_len=10,
    lr=1e-3,
):
    if name not in experiment_data:
        experiment_data[name] = {
            "metrics": {"train": [], "val": [], "test": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "history": {"val_top3": []},
            "config": {},
        }
    print(f"\n=== Dataset: {name} | Ablation: {ablation_type} ===")
    # Limit cases for runtime
    try:
        starts_all = (
            df.sort_values("timestamp")
            .groupby("case_id")["timestamp"]
            .min()
            .reset_index()
        )
        if len(starts_all) > 4000:
            keep_cases = set(starts_all.iloc[:4000]["case_id"])
            df = df[df["case_id"].isin(keep_cases)].copy()
    except Exception:
        pass
    # Time-based split (no leakage)
    train_cases, val_cases, test_cases = time_based_split(df, 0.7, 0.15)
    df_train = df[df["case_id"].isin(train_cases)].copy()
    df_val = df[df["case_id"].isin(val_cases)].copy()
    df_test = df[df["case_id"].isin(test_cases)].copy()

    # Build vocab on train (and optionally collapse rare)
    if ablation_type == "rare_activity_collapse":
        train_mask = df["case_id"].isin(train_cases).to_numpy()
        act2id, id2act, pad_id, unk_id = build_vocab_from_df(
            df, use_unk=True, rare_threshold=unk_threshold, train_mask=train_mask
        )
    else:
        act2id, id2act, pad_id, unk_id = build_vocab_from_df(df_train, use_unk=False)

    # Build prefix samples
    samples_train = build_prefix_dataset_with_vocab(
        df_train, act2id, unk_id, pad_id, max_prefix_len, 1
    )
    samples_val = build_prefix_dataset_with_vocab(
        df_val, act2id, unk_id, pad_id, max_prefix_len, 1
    )
    samples_test = build_prefix_dataset_with_vocab(
        df_test, act2id, unk_id, pad_id, max_prefix_len, 1
    )

    # Re-normalize time features based on train only (CRITICAL normalization)
    if len(samples_train) > 0:
        concat_feats = [
            s["seq_feats"] for s in samples_train if s["seq_feats"].shape[0] > 0
        ]
        if len(concat_feats) > 0:
            all_feats = np.concatenate(concat_feats, axis=0)
            dt_mean, dt_std = all_feats[:, 0].mean(), all_feats[:, 0].std() + 1e-6
            ss_mean, ss_std = all_feats[:, 1].mean(), all_feats[:, 1].std() + 1e-6

            def norm_samples(samples):
                for s in samples:
                    if s["seq_feats"].shape[0] > 0:
                        s["seq_feats"][:, 0] = (s["seq_feats"][:, 0] - dt_mean) / dt_std
                        s["seq_feats"][:, 1] = (s["seq_feats"][:, 1] - ss_mean) / ss_std

            norm_samples(samples_train)
            norm_samples(samples_val)
            norm_samples(samples_test)

    vocab_size = len(act2id) + (1 if unk_id is not None else 0)
    print(
        f"Samples train/val/test: {len(samples_train)}/{len(samples_val)}/{len(samples_test)}; vocab={vocab_size} (unk={'on' if unk_id is not None else 'off'})"
    )
    if len(samples_train) == 0 or vocab_size < 2:
        print("Not enough data to train. Skipping.")
        return

    # DataLoaders
    ds_train = PrefixDataset(
        samples_train, pad_id=0, max_len=max_prefix_len, num_cont=5
    )
    ds_val = PrefixDataset(samples_val, pad_id=0, max_len=max_prefix_len, num_cont=5)
    ds_test = PrefixDataset(samples_test, pad_id=0, max_len=max_prefix_len, num_cont=5)
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

    # Model and optimizer (move to device BEFORE optimizer)
    model = LSTMBaseline(
        vocab_size=vocab_size,
        emb_dim=64,
        cont_dim=5,
        hidden=128,
        num_layers=1,
        pad_idx=0,
    ).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Config record
    experiment_data[name]["config"][ablation_type] = {
        "max_prefix_len": max_prefix_len,
        "batch_size": batch_size,
        "lr": lr,
        "vocab_size": vocab_size,
        "unk_threshold": (
            unk_threshold if ablation_type == "rare_activity_collapse" else None
        ),
        "epochs": max_epochs,
    }

    # Training loop with validation tracking (CRITICAL EVALUATION REQUIREMENTS)
    best_val_top3 = -1.0
    best_state = None
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
            model, dl_val, criterion, device, vocab_size, 0
        )
        print(
            f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | val_top3={val_top3:.4f}"
        )
        experiment_data[name]["losses"]["train"].append(
            (ablation_type, epoch, float(train_loss))
        )
        experiment_data[name]["losses"]["val"].append(
            (ablation_type, epoch, float(val_loss))
        )
        experiment_data[name]["metrics"]["val"].append(
            (
                ablation_type,
                epoch,
                {
                    "acc": float(val_acc),
                    "macro_f1": float(val_f1),
                    "top3": float(val_top3),
                },
            )
        )
        experiment_data[name]["history"]["val_top3"].append(
            (ablation_type, epoch, float(val_top3))
        )
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
        model, dl_train, criterion, device, vocab_size, 0
    )
    val_loss, val_acc, val_f1, val_top3, _, _, _ = evaluate(
        model, dl_val, criterion, device, vocab_size, 0
    )
    test_loss, test_acc, test_f1, test_top3, y_true_t, y_pred_t, probs_t = evaluate(
        model, dl_test, criterion, device, vocab_size, 0
    )
    print(
        f"[{name} | {ablation_type}] Train: loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} top3={train_top3:.4f}"
    )
    print(
        f"[{name} | {ablation_type}] Test:  loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f} top3={test_top3:.4f}"
    )

    experiment_data[name]["metrics"]["train"].append(
        (
            ablation_type,
            "final",
            {
                "loss": float(train_loss),
                "acc": float(train_acc),
                "macro_f1": float(train_f1),
                "top3": float(train_top3),
            },
        )
    )
    experiment_data[name]["metrics"]["val"].append(
        (
            ablation_type,
            "final",
            {
                "loss": float(val_loss),
                "acc": float(val_acc),
                "macro_f1": float(val_f1),
                "top3": float(val_top3),
            },
        )
    )
    experiment_data[name]["metrics"]["test"].append(
        (
            ablation_type,
            "final",
            {
                "loss": float(test_loss),
                "acc": float(test_acc),
                "macro_f1": float(test_f1),
                "top3": float(test_top3),
            },
        )
    )
    experiment_data[name]["predictions"] = y_pred_t.tolist()
    experiment_data[name]["ground_truth"] = y_true_t.tolist()
    experiment_data[name]["probs_shape"] = list(probs_t.shape)


def run():
    # Load datasets via pm4py from local XES
    datasets = load_datasets()
    # For each dataset, prepare and run baseline and ablation
    for key, df in datasets.items():
        df = df.copy()
        df["activity"] = df["activity"].astype(str)
        df["case_id"] = df["case_id"].astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        # Baseline
        train_one_dataset(
            key,
            df,
            ablation_type="baseline",
            unk_threshold=None,
            max_epochs=6,
            batch_size=128,
            max_prefix_len=10,
            lr=1e-3,
        )
        # Rare activity collapsing ablation
        train_one_dataset(
            key,
            df,
            ablation_type="rare_activity_collapse",
            unk_threshold=5,
            max_epochs=6,
            batch_size=128,
            max_prefix_len=10,
            lr=1e-3,
        )
    # Save all metrics
    np.save(
        os.path.join(working_dir, "experiment_data.npy"),
        experiment_data,
        allow_pickle=True,
    )
    print(
        f"[save] experiment_data saved to {os.path.join(working_dir, 'experiment_data.npy')}"
    )


# Execute immediately (no __main__ guard to avoid non-execution in some environments)
run()
