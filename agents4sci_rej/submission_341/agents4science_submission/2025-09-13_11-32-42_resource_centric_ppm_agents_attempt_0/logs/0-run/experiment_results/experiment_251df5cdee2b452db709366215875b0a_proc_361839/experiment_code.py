import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
from datetime import datetime

# Device handling (mandatory)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ---------------- Robust XES discovery & loading ----------------
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
        raise ImportError("pm4py is required. Install via `pip install pm4py`.") from e
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
    df = df[["case_id", "activity", "lifecycle", "timestamp", "resource"]]
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    return df


def load_datasets() -> Dict[str, pd.DataFrame]:
    data_dir = _resolve_data_dir()
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
                print(
                    f"[data] Loaded {key}: events={len(loaded[key])}, cases={loaded[key]['case_id'].nunique()}"
                )
            except Exception as e:
                print(f"[warn] Failed to load {key} from {path}: {e}")
        else:
            print(f"[data] Not found for {key} (patterns {pats})")
    if not loaded:
        # Fallback to any .xes files with generic names
        any_files = list(data_dir.glob("*.xes")) + list(data_dir.glob("*.xes.gz"))
        for p in any_files:
            key = p.stem
            try:
                loaded[key] = xes_to_df(p)
            except Exception as e:
                print(f"[warn] Failed to load {p}: {e}")
    return loaded


# ---------------- Prefix building and splits ----------------
def build_prefix_dataset(df, max_prefix_len=15, min_prefix_len=1):
    df = df.copy()
    if "lifecycle" in df.columns:
        # keep completes if present; otherwise keep all
        lc_mask = df["lifecycle"].astype(str).str.lower().eq("complete")
        if lc_mask.any():
            df = df[lc_mask]
    df = df.sort_values(["case_id", "timestamp"])
    acts = df["activity"].astype(str).unique().tolist()
    act2id = {a: i + 1 for i, a in enumerate(sorted(acts))}  # 0 for PAD
    id2act = {i: a for a, i in act2id.items()}
    pad_id = 0
    samples = []
    for cid, g in df.groupby("case_id", sort=False):
        g = g.sort_values("timestamp")
        ts = (
            pd.to_datetime(g["timestamp"], utc=True).astype("int64").to_numpy() // 10**9
        )
        acts_ids = np.array(
            [act2id[a] for a in g["activity"].astype(str)], dtype=np.int64
        )
        T = len(acts_ids)
        if T < 2:
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
        max_k = min(max_prefix_len, T - 1)
        for k in range(min_prefix_len, max_k + 1):
            samples.append(
                {
                    "case_id": str(cid),
                    "seq_acts": acts_ids[:k].tolist(),
                    "seq_feats": feats[:k].copy(),
                    "target": int(acts_ids[k]),
                }
            )
    # Normalize time deltas and since_start using global (temporary) stats; will re-norm by train stats later
    if len(samples) > 0:
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


def time_based_split(df, train_frac=0.7, val_frac=0.15):
    starts = (
        df.groupby("case_id", as_index=False)["timestamp"]
        .min()
        .sort_values("timestamp")
    )
    n = len(starts)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_cases = set(starts.iloc[:n_train]["case_id"].astype(str))
    val_cases = set(starts.iloc[n_train : n_train + n_val]["case_id"].astype(str))
    test_cases = set(starts.iloc[n_train + n_val :]["case_id"].astype(str))
    return train_cases, val_cases, test_cases


class PrefixDataset(Dataset):
    def __init__(self, samples, pad_id, max_len=15, num_cont=5):
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


# ---------------- Model ----------------
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
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden, vocab_size + 1)  # includes PAD
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


# ---------------- Metrics ----------------
def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    # probs: (N, C), y_true: (N,)
    if probs.size == 0 or y_true.size == 0:
        return 0.0
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (
            confidences <= bins[i + 1] if i < n_bins - 1 else confidences <= bins[i + 1]
        )
        if not np.any(mask):
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (np.sum(mask) / len(confidences)) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate(model, loader, criterion, device, num_classes, pad_idx):
    model.eval()
    total_loss = 0.0
    ys, preds_top1, probs_list = [], [], []
    top3_correct = 0
    n_total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["acts"], batch["feats"], batch["mask"])
            # Exclude PAD from being predicted by setting its logit very low
            logits[:, pad_idx] = -1e9
            loss = criterion(logits, batch["y"])
            total_loss += loss.item() * logits.size(0)
            probs = torch.softmax(logits, dim=1)
            top1 = torch.argmax(probs, dim=1)
            k_val = min(3, probs.size(1))
            _, topk_idx = torch.topk(probs, k=k_val, dim=1)
            ys.extend(batch["y"].detach().cpu().tolist())
            preds_top1.extend(top1.detach().cpu().tolist())
            probs_list.append(probs.detach().cpu().numpy())
            for i in range(batch["y"].size(0)):
                if batch["y"][i].item() in topk_idx[i].detach().cpu().tolist():
                    top3_correct += 1
            n_total += batch["y"].size(0)
    avg_loss = total_loss / max(1, n_total)
    y_true = np.array(ys, dtype=np.int64)
    y_pred = np.array(preds_top1, dtype=np.int64)
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    try:
        f1 = float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        f1 = 0.0
    top3 = float(top3_correct / max(1, n_total))
    probs_concat = (
        np.concatenate(probs_list, axis=0)
        if len(probs_list) > 0
        else np.zeros((0, num_classes + 1))
    )
    ece = compute_ece(probs_concat, y_true, n_bins=10)
    return avg_loss, acc, f1, top3, ece, y_true, y_pred, probs_concat


# ---------------- Training & Tuning ----------------
def train_with_epoch_tuning(
    name,
    df,
    base_epochs=40,
    maybe_extend_to=60,
    batch_size=256,
    max_prefix_len=15,
    lr=3e-4,
):
    print(f"\n=== Dataset: {name} ===")
    # Time-based split
    train_cases, val_cases, test_cases = time_based_split(df, 0.7, 0.15)
    # Build samples and vocab
    samples_all, act2id, id2act, pad_id = build_prefix_dataset(
        df, max_prefix_len=max_prefix_len
    )
    # Filter splits
    samples_train = [s for s in samples_all if s["case_id"] in train_cases]
    samples_val = [s for s in samples_all if s["case_id"] in val_cases]
    samples_test = [s for s in samples_all if s["case_id"] in test_cases]
    # Recompute normalization using train samples only (no leakage)
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
    print(
        f"Samples train/val/test: {len(samples_train)}/{len(samples_val)}/{len(samples_test)}; vocab={len(act2id)}"
    )
    if len(samples_train) == 0 or len(act2id) < 2:
        print("Not enough data to train. Skipping.")
        return None

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

    # Training with early stopping and potential extension
    best_val_top3 = -1.0
    best_state = None
    best_epoch = 0
    hist = {
        "train_loss": [],
        "val_loss": [],
        "val_top3": [],
        "val_acc": [],
        "val_f1": [],
        "val_ece": [],
    }
    patience = 8
    no_improve = 0

    def train_epochs(n_epochs, start_epoch_idx=0):
        nonlocal best_val_top3, best_state, best_epoch, no_improve
        epochs_run = 0
        for e in range(1, n_epochs + 1):
            epoch = start_epoch_idx + e
            model.train()
            total = 0
            running_loss = 0.0
            for batch in dl_train:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                optimizer.zero_grad()
                logits = model(batch["acts"], batch["feats"], batch["mask"])
                logits[:, pad_id] = -1e9  # mask PAD from being predicted
                loss = criterion(logits, batch["y"])
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * logits.size(0)
                total += logits.size(0)
            train_loss = running_loss / max(1, total)
            val_loss, val_acc, val_f1, val_top3, val_ece, _, _, _ = evaluate(
                model, dl_val, criterion, device, len(act2id), pad_id
            )
            print(
                f"Epoch {epoch}: validation_loss = {val_loss:.4f} | acc={val_acc:.4f} | f1={val_f1:.4f} | top3={val_top3:.4f} | ece={val_ece:.4f}"
            )
            hist["train_loss"].append(train_loss)
            hist["val_loss"].append(val_loss)
            hist["val_top3"].append(val_top3)
            hist["val_acc"].append(val_acc)
            hist["val_f1"].append(val_f1)
            hist["val_ece"].append(val_ece)
            if val_top3 > best_val_top3 + 1e-6:
                best_val_top3 = val_top3
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping (no improvement in {patience} epochs).")
                break
            epochs_run += 1
        return start_epoch_idx + epochs_run

    last_epoch = train_epochs(base_epochs, start_epoch_idx=0)

    def should_extend(val_top3_hist, base_epochs):
        if len(val_top3_hist) < min(10, base_epochs):
            return False
        last5 = val_top3_hist[-5:]
        prev5 = val_top3_hist[-10:-5]
        if len(prev5) < 5:
            return False
        mean_gain = np.mean(last5) - np.mean(prev5)
        condition = (best_epoch >= max(1, base_epochs - 3)) and (mean_gain > 0.002)
        return condition

    if should_extend(hist["val_top3"], base_epochs) and maybe_extend_to > len(
        hist["train_loss"]
    ):
        print(
            f"Extending training to {maybe_extend_to} epochs due to improving validation top-3."
        )
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)
        # reset patience
        nonlocal_no_improve = 0  # local variable to avoid interference; but we will reuse no_improve variable
        no_improve = 0
        extra_epochs = maybe_extend_to - len(hist["train_loss"])
        if extra_epochs > 0:
            last_epoch = train_epochs(
                extra_epochs, start_epoch_idx=len(hist["train_loss"])
            )

    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Final evaluations
    train_loss, train_acc, train_f1, train_top3, train_ece, _, _, _ = evaluate(
        model, dl_train, criterion, device, len(act2id), pad_id
    )
    val_loss, val_acc, val_f1, val_top3, val_ece, _, _, _ = evaluate(
        model, dl_val, criterion, device, len(act2id), pad_id
    )
    test_loss, test_acc, test_f1, test_top3, test_ece, y_true_t, y_pred_t, probs_t = (
        evaluate(model, dl_test, criterion, device, len(act2id), pad_id)
    )

    print(
        f"[{name}] Train: loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} top3={train_top3:.4f} ece={train_ece:.4f}"
    )
    print(
        f"[{name}] Val:   loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} top3={val_top3:.4f} ece={val_ece:.4f}"
    )
    print(
        f"[{name}] Test:  loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f} top3={test_top3:.4f} ece={test_ece:.4f}"
    )

    # Prepare results dict for this dataset
    results = {
        "metrics": {
            "train": [
                (
                    "final",
                    {
                        "loss": train_loss,
                        "acc": train_acc,
                        "macro_f1": train_f1,
                        "top3": train_top3,
                        "ece": train_ece,
                    },
                )
            ],
            "val": [
                (
                    "final",
                    {
                        "loss": val_loss,
                        "acc": val_acc,
                        "macro_f1": val_f1,
                        "top3": val_top3,
                        "ece": val_ece,
                    },
                )
            ],
            "test": [
                (
                    "final",
                    {
                        "loss": test_loss,
                        "acc": test_acc,
                        "macro_f1": test_f1,
                        "top3": test_top3,
                        "ece": test_ece,
                    },
                )
            ],
        },
        "losses": {
            "train": list(enumerate(hist["train_loss"], start=1)),
            "val": list(enumerate(hist["val_loss"], start=1)),
        },
        "val_curves": {
            "acc": list(enumerate(hist["val_acc"], start=1)),
            "f1": list(enumerate(hist["val_f1"], start=1)),
            "top3": list(enumerate(hist["val_top3"], start=1)),
            "ece": list(enumerate(hist["val_ece"], start=1)),
        },
        "predictions": y_pred_t.tolist(),
        "ground_truth": y_true_t.tolist(),
        "epochs_trained": len(hist["train_loss"]),
        "best_val_top3": best_val_top3,
        "best_epoch": best_epoch,
        "vocab_size": len(act2id),
        "pad_idx": pad_id,
    }

    # Save arrays
    base = os.path.join(working_dir, name)
    np.save(
        os.path.join(working_dir, f"{name}_train_loss.npy"),
        np.array(hist["train_loss"], dtype=np.float32),
    )
    np.save(
        os.path.join(working_dir, f"{name}_val_loss.npy"),
        np.array(hist["val_loss"], dtype=np.float32),
    )
    np.save(
        os.path.join(working_dir, f"{name}_val_top3.npy"),
        np.array(hist["val_top3"], dtype=np.float32),
    )
    np.save(
        os.path.join(working_dir, f"{name}_val_acc.npy"),
        np.array(hist["val_acc"], dtype=np.float32),
    )
    np.save(
        os.path.join(working_dir, f"{name}_val_f1.npy"),
        np.array(hist["val_f1"], dtype=np.float32),
    )
    np.save(
        os.path.join(working_dir, f"{name}_val_ece.npy"),
        np.array(hist["val_ece"], dtype=np.float32),
    )
    np.save(
        os.path.join(working_dir, f"{name}_y_true.npy"),
        np.array(y_true_t, dtype=np.int64),
    )
    np.save(
        os.path.join(working_dir, f"{name}_y_pred.npy"),
        np.array(y_pred_t, dtype=np.int64),
    )
    try:
        cm = confusion_matrix(y_true_t, y_pred_t)
        np.save(os.path.join(working_dir, f"{name}_confusion_matrix.npy"), cm)
    except Exception as e:
        print(f"[warn] Confusion matrix failed: {e}")
    return results


# ---------------- Experiment Orchestration ----------------
experiment_data = {}


def main():
    global experiment_data
    datasets = load_datasets()
    if len(datasets) == 0:
        print("No datasets found. Exiting after saving empty experiment_data.")
        np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
        return
    # Ensure keys present for BPI2017 and ROAD if available; limit large datasets by earliest 5000 cases for runtime
    for key, df in datasets.items():
        try:
            starts = (
                df.groupby("case_id", as_index=False)["timestamp"]
                .min()
                .sort_values("timestamp")
            )
            if len(starts) > 5000:
                keep_cases = set(starts.iloc[:5000]["case_id"])
                df_small = df[df["case_id"].isin(keep_cases)].copy()
            else:
                df_small = df
        except Exception:
            df_small = df
        res = train_with_epoch_tuning(
            key,
            df_small,
            base_epochs=40,
            maybe_extend_to=60,
            batch_size=256,
            max_prefix_len=15,
            lr=3e-4,
        )
        if res is not None:
            experiment_data[key] = res
            # also append per-epoch metrics to experiment_data for protocol
            # Already contained in res under losses and val_curves
    # Save consolidated experiment data
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


# Execute immediately (no __main__ guard)
main()
