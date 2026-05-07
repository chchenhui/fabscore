import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pandas as pd
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Device setup (must be at start)
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

# Global experiment data container per dataset with nested ablations
experiment_data = {}


# --------------- Robust XES discovery and loading (from workspace input/) ---------------
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
        f"Could not locate a directory containing .xes files.\nChecked:{tried}"
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
        p = _first_match(data_dir, pats)
        if p is not None:
            try:
                df = xes_to_df(p)
                if len(df) > 0:
                    loaded[key] = df
            except Exception as e:
                print(f"[warn] Failed to load {key} from {p}: {e}")
        else:
            print(f"[data] Not found for {key} (patterns={pats})")
    if not loaded:
        raise FileNotFoundError(
            "No known XES files found under discovered directories."
        )
    print(f"[data] Loaded datasets: {list(loaded.keys())}")
    return loaded


# --------------- Time-based split (by case start time) ---------------
def time_based_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
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


# --------------- Vocabulary building ---------------
def build_vocab_from_df(
    df: pd.DataFrame, use_unk: bool, rare_threshold: int, train_cases: Optional[set]
):
    pad_id = 0
    if not use_unk:
        acts = sorted(df["activity"].astype(str).unique().tolist())
        act2id = {a: i + 1 for i, a in enumerate(acts)}
        unk_id = None
    else:
        mask = (
            df["case_id"].isin(train_cases)
            if train_cases is not None
            else pd.Series(True, index=df.index)
        )
        counts = df.loc[mask, "activity"].astype(str).value_counts()
        frequent = sorted(counts[counts >= rare_threshold].index.tolist())
        act2id = {a: i + 1 for i, a in enumerate(frequent)}
        unk_id = len(act2id) + 1
    id2act = {i: a for a, i in act2id.items()}
    if use_unk:
        id2act[unk_id] = "UNK"
    vocab_size = max([*id2act.keys(), (unk_id if use_unk else 0)]) + 1
    return act2id, id2act, pad_id, unk_id, vocab_size


# --------------- Prefix sample construction with correct alignment ---------------
def build_prefix_dataset_with_vocab(
    df: pd.DataFrame,
    act2id: dict,
    unk_id: Optional[int],
    pad_id: int,
    max_prefix_len=10,
    min_prefix_len=1,
):
    df = df.copy()
    if "lifecycle" in df.columns:
        mask = df["lifecycle"].astype(str).str.lower().eq("complete")
        if mask.any():
            df = df[mask]
    df = df.sort_values(["case_id", "timestamp"])
    samples = []
    for cid, g in df.groupby("case_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        acts = g["activity"].astype(str).tolist()
        ts = pd.to_datetime(g["timestamp"], utc=True).view("int64") // 10**9
        kept_ids, kept_idx = [], []
        for i, a in enumerate(acts):
            if a in act2id:
                kept_ids.append(act2id[a])
                kept_idx.append(i)
            elif unk_id is not None:
                kept_ids.append(unk_id)
                kept_idx.append(i)
            else:
                continue
        if len(kept_ids) < 2:
            continue
        kept_ids = np.asarray(kept_ids, dtype=np.int64)
        kept_idx = np.asarray(kept_idx, dtype=np.int64)
        ts_kept = ts.iloc[kept_idx].astype(np.int64).to_numpy()
        g_ts = pd.to_datetime(g["timestamp"].iloc[kept_idx], utc=True)
        hours = (g_ts.dt.hour.to_numpy(dtype=float) / 23.0).astype(np.float32)
        weekdays = (g_ts.dt.weekday.to_numpy(dtype=float) / 6.0).astype(np.float32)
        working = (
            (g_ts.dt.weekday.to_numpy() < 5)
            & (g_ts.dt.hour.to_numpy() >= 8)
            & (g_ts.dt.hour.to_numpy() <= 17)
        ).astype(np.float32)
        deltas = np.diff(ts_kept, prepend=ts_kept[0]).astype(np.float32)
        since_start = (ts_kept - ts_kept[0]).astype(np.float32)
        feats = np.stack(
            [deltas, since_start, hours, weekdays, working], axis=1
        ).astype(np.float32)
        T = len(kept_ids)
        max_k = min(max_prefix_len, T - 1)
        for k in range(min_prefix_len, max_k + 1):
            seq_acts = kept_ids[:k].tolist()
            seq_feats = feats[:k]
            target = int(kept_ids[k])
            samples.append(
                {
                    "case_id": cid,
                    "seq_acts": seq_acts,
                    "seq_feats": seq_feats.copy(),
                    "target": target,
                }
            )
    if len(samples) > 0:
        all_feats = np.concatenate(
            [s["seq_feats"] for s in samples if len(s["seq_feats"]) > 0], axis=0
        )
        dt_mean, dt_std = all_feats[:, 0].mean(), all_feats[:, 0].std() + 1e-6
        ss_mean, ss_std = all_feats[:, 1].mean(), all_feats[:, 1].std() + 1e-6
        for s in samples:
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
            "y": torch.tensor(int(s["target"]), dtype=torch.long),
        }


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


# --------------- Baseline model ---------------
class LSTMBaseline(nn.Module):
    def __init__(
        self, vocab_size, emb_dim=64, cont_dim=5, hidden=128, num_layers=1, pad_idx=0
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim + cont_dim,
            hidden_size=hidden,
            batch_first=True,
            num_layers=num_layers,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden, vocab_size)
        self.pad_idx = pad_idx

    def forward(self, acts, feats, mask):
        x = self.emb(acts)
        x = torch.cat([x, feats], dim=-1)
        out, (h, c) = self.lstm(x)
        h_last = self.dropout(h[-1])
        logits = self.fc(h_last)
        return logits


# --------------- Evaluation ---------------
def evaluate(model, loader, criterion, device, num_classes, pad_idx):
    model.eval()
    total_loss = 0.0
    ys, preds_top1 = [], []
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
            for i in range(batch["y"].size(0)):
                if int(batch["y"][i].item()) in topk_idx[i].detach().cpu().tolist():
                    top3_correct += 1
            n_total += batch["y"].size(0)
    avg_loss = total_loss / max(1, n_total)
    y_true = np.array(ys)
    y_pred = np.array(preds_top1)
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    try:
        f1 = float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        f1 = 0.0
    top3 = float(top3_correct / max(1, n_total))
    return avg_loss, acc, f1, top3


# --------------- Training per dataset/ablation ---------------
def train_one_dataset(
    name,
    df,
    ablation_type="baseline",
    unk_threshold=5,
    max_epochs=5,
    batch_size=128,
    max_prefix_len=10,
    lr=1e-3,
):
    print(f"\n=== Dataset: {name} | Ablation: {ablation_type} ===")
    starts_all = (
        df.sort_values("timestamp")
        .groupby("case_id")["timestamp"]
        .min()
        .reset_index()
        .sort_values("timestamp")
    )
    if len(starts_all) > 3000:
        keep_cases = set(starts_all.iloc[:3000]["case_id"])
        df = df[df["case_id"].isin(keep_cases)].copy()
    train_cases, val_cases, test_cases = time_based_split(df, 0.7, 0.15)
    df_train = df[df["case_id"].isin(train_cases)].copy()
    df_val = df[df["case_id"].isin(val_cases)].copy()
    df_test = df[df["case_id"].isin(test_cases)].copy()
    use_unk = ablation_type == "rare_activity_collapse"
    act2id, id2act, pad_id, unk_id, vocab_size = build_vocab_from_df(
        df,
        use_unk=use_unk,
        rare_threshold=(unk_threshold if use_unk else 0),
        train_cases=train_cases,
    )
    samples_train = build_prefix_dataset_with_vocab(
        df_train, act2id, unk_id, pad_id, max_prefix_len=max_prefix_len
    )
    samples_val = build_prefix_dataset_with_vocab(
        df_val, act2id, unk_id, pad_id, max_prefix_len=max_prefix_len
    )
    samples_test = build_prefix_dataset_with_vocab(
        df_test, act2id, unk_id, pad_id, max_prefix_len=max_prefix_len
    )
    if len(samples_train) > 0:
        all_feats = np.concatenate(
            [s["seq_feats"] for s in samples_train if s["seq_feats"].shape[0] > 0],
            axis=0,
        )
        dt_mean, dt_std = all_feats[:, 0].mean(), all_feats[:, 0].std() + 1e-6
        ss_mean, ss_std = all_feats[:, 1].mean(), all_feats[:, 1].std() + 1e-6

        def renorm(samples):
            for s in samples:
                s["seq_feats"][:, 0] = (s["seq_feats"][:, 0] - dt_mean) / dt_std
                s["seq_feats"][:, 1] = (s["seq_feats"][:, 1] - ss_mean) / ss_std

        renorm(samples_train)
        renorm(samples_val)
        renorm(samples_test)
    print(
        f"Samples train/val/test: {len(samples_train)}/{len(samples_val)}/{len(samples_test)}; vocab_size={vocab_size} (UNK={'on' if unk_id is not None else 'off'})"
    )
    if len(samples_train) == 0 or vocab_size < 3:
        print("Insufficient data. Skipping.")
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
        vocab_size=vocab_size,
        emb_dim=64,
        cont_dim=5,
        hidden=128,
        num_layers=1,
        pad_idx=pad_id,
    ).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if name not in experiment_data:
        experiment_data[name] = {}
    if ablation_type not in experiment_data[name]:
        experiment_data[name][ablation_type] = {
            "metrics": {"train": [], "val": [], "test": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "history": {"epochs": [], "val_top3": [], "val_loss": []},
            "config": {
                "ablation": ablation_type,
                "unk_threshold": (unk_threshold if use_unk else None),
                "max_prefix_len": max_prefix_len,
                "batch_size": batch_size,
                "lr": lr,
                "vocab_size": vocab_size,
            },
        }
    best_val_top3 = -1.0
    best_state = None
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
        val_loss, val_acc, val_f1, val_top3 = evaluate(
            model, dl_val, criterion, device, vocab_size, pad_id
        )
        print(
            f"Epoch {epoch}: validation_loss = {val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | val_top3={val_top3:.4f}"
        )
        experiment_data[name][ablation_type]["losses"]["train"].append(
            (epoch, train_loss)
        )
        experiment_data[name][ablation_type]["losses"]["val"].append((epoch, val_loss))
        experiment_data[name][ablation_type]["metrics"]["val"].append(
            (epoch, {"acc": val_acc, "macro_f1": val_f1, "top3": val_top3})
        )
        experiment_data[name][ablation_type]["history"]["epochs"].append(epoch)
        experiment_data[name][ablation_type]["history"]["val_top3"].append(val_top3)
        experiment_data[name][ablation_type]["history"]["val_loss"].append(val_loss)
        if val_top3 > best_val_top3:
            best_val_top3 = val_top3
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    train_loss, train_acc, train_f1, train_top3 = evaluate(
        model, dl_train, criterion, device, vocab_size, pad_id
    )
    val_loss, val_acc, val_f1, val_top3 = evaluate(
        model, dl_val, criterion, device, vocab_size, pad_id
    )
    test_loss, test_acc, test_f1, test_top3 = evaluate(
        model, dl_test, criterion, device, vocab_size, pad_id
    )
    print(
        f"[{name} | {ablation_type}] Train: loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} top3={train_top3:.4f}"
    )
    print(
        f"[{name} | {ablation_type}] Test:  loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f} top3={test_top3:.4f}"
    )
    experiment_data[name][ablation_type]["metrics"]["train"].append(
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
    experiment_data[name][ablation_type]["metrics"]["val"].append(
        (
            "final",
            {"loss": val_loss, "acc": val_acc, "macro_f1": val_f1, "top3": val_top3},
        )
    )
    experiment_data[name][ablation_type]["metrics"]["test"].append(
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
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for batch in dl_test:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            logits = model(batch["acts"], batch["feats"], batch["mask"])
            probs = torch.softmax(logits, dim=1)
            top1 = torch.argmax(probs, dim=1)
            y_true_all.extend(batch["y"].detach().cpu().tolist())
            y_pred_all.extend(top1.detach().cpu().tolist())
    experiment_data[name][ablation_type]["predictions"] = y_pred_all
    experiment_data[name][ablation_type]["ground_truth"] = y_true_all


def run():
    datasets = load_datasets()
    for key, df in datasets.items():
        df = df.copy()
        df["activity"] = df["activity"].astype(str)
        df["case_id"] = df["case_id"].astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values(["timestamp", "case_id"])
        train_one_dataset(
            key,
            df,
            ablation_type="baseline",
            unk_threshold=0,
            max_epochs=5,
            batch_size=128,
            max_prefix_len=10,
            lr=1e-3,
        )
        train_one_dataset(
            key,
            df,
            ablation_type="rare_activity_collapse",
            unk_threshold=5,
            max_epochs=5,
            batch_size=128,
            max_prefix_len=10,
            lr=1e-3,
        )
    np.save(
        os.path.join(working_dir, "experiment_data.npy"),
        experiment_data,
        allow_pickle=True,
    )
    print(
        f"[save] experiment_data saved to {os.path.join(working_dir, 'experiment_data.npy')}"
    )


run()

# --------- Load and Plot from experiment_data.npy (Next-Activity BPM visualizations) ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Plot per dataset and ablation
for dataset_name, ablations in experiment_data.items():
    for ablation, payload in ablations.items():
        # Training/Validation loss curves
        try:
            plt.figure()
            train_hist = payload.get("losses", {}).get("train", [])
            val_hist = payload.get("losses", {}).get("val", [])
            if len(train_hist) > 0:
                epochs, tloss = zip(*train_hist)
                plt.plot(epochs, tloss, label="Train Loss")
            if len(val_hist) > 0:
                epochs_v, vloss = zip(*val_hist)
                plt.plot(epochs_v, vloss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{dataset_name} - {ablation} | Training/Validation Loss (Next-Activity)"
            )
            plt.legend()
            fname = os.path.join(
                working_dir, f"{dataset_name}_{ablation}_loss_curves.png"
            )
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating loss curves for {dataset_name}-{ablation}: {e}")
            plt.close()
        # Validation Top-3 over epochs
        try:
            plt.figure()
            hist = payload.get("history", {})
            epochs = hist.get("epochs", [])
            val_top3 = hist.get("val_top3", [])
            if len(epochs) > 0 and len(val_top3) > 0:
                plt.plot(epochs, val_top3, marker="o", label="Val Top-3")
                plt.xlabel("Epoch")
                plt.ylabel("Top-3 Accuracy")
                plt.title(
                    f"{dataset_name} - {ablation} | Validation Top-3 (Next-Activity)"
                )
                plt.legend()
                fname = os.path.join(
                    working_dir, f"{dataset_name}_{ablation}_val_top3.png"
                )
                plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating val top-3 plot for {dataset_name}-{ablation}: {e}")
            plt.close()
        # Confusion matrix on test
        try:
            y_true = payload.get("ground_truth", [])
            y_pred = payload.get("predictions", [])
            if len(y_true) > 0 and len(y_pred) == len(y_true):
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(6, 5))
                plt.imshow(cm, interpolation="nearest", cmap="Blues")
                plt.colorbar()
                plt.title(
                    f"{dataset_name} - {ablation} | Confusion Matrix (Next-Activity)\nLeft: Ground Truth, Right: Predicted Labels"
                )
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.tight_layout()
                fname = os.path.join(
                    working_dir, f"{dataset_name}_{ablation}_confusion_matrix.png"
                )
                plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {dataset_name}-{ablation}: {e}")
            plt.close()

# --------- Print concise BPM metrics summary from saved data ----------
for dataset_name, ablations in experiment_data.items():
    for ablation, payload in ablations.items():
        tests = payload.get("metrics", {}).get("test", [])
        final = None
        for tag, metric in tests:
            if tag == "final":
                final = metric
        if final is not None:
            print(
                f"[SUMMARY] {dataset_name} | {ablation} | Test: acc={final.get('acc',0):.4f} | macro_f1={final.get('macro_f1',0):.4f} | top3={final.get('top3',0):.4f}"
            )
