import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
import random
from pathlib import Path
import math
from typing import Dict, List, Optional, Tuple

# Device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


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
        "Tips:\n  • Ensure filenames include BPI 2012/2017 or 'Road_Traffic_Fine_Management_Process' for auto-match."
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
        raise ImportError("pm4py is required. Install: pip install pm4py") from e
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
    patterns = {
        "BPI2012": ["BPI_Challenge_2012*.xes*", "BPI2012*.xes*", "*2012*.xes*"],
        "BPI2017": ["BPI_Challenge_2017*.xes*", "BPI2017*.xes*", "*2017*.xes*"],
        "ROAD": [
            "Road_Traffic_Fine_Management_Process*.xes*",
            "*Traffic*Fine*.xes*",
            "*Traffic*.xes*",
        ],
    }
    datasets: Dict[str, pd.DataFrame] = {}
    try:
        data_dir = _resolve_data_dir()
        available = sorted(
            [
                p.name
                for p in list(data_dir.glob("*.xes")) + list(data_dir.glob("*.xes.gz"))
            ]
        )
        print(f"[data] Available in {data_dir}: {available}")
        for key, pats in patterns.items():
            p = _first_match(data_dir, pats)
            if p is not None:
                try:
                    datasets[key] = xes_to_df(p)
                    print(f"[data] Loaded {key}: {p.name}, events={len(datasets[key])}")
                except Exception as e:
                    print(f"[warn] Failed to load {key} from {p}: {e}")
            else:
                print(f"[data] Not found for {key} (patterns {patterns[key]})")
    except Exception as e:
        print(f"[warn] Dataset discovery failed: {e}")
    return datasets


def make_synthetic_df(n_cases=500, max_events=15, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    activities = ["A", "B", "C", "D", "E", "F", "G"]
    rows = []
    base_time = int(pd.Timestamp("2020-01-01", tz="UTC").value // 10**9)
    for c in range(n_cases):
        T = int(rng.integers(3, max_events))
        start = base_time + int(rng.integers(0, 60 * 60 * 24 * 30))
        ts = np.cumsum(rng.integers(10, 3600, size=T)) + start
        for t in ts:
            rows.append(
                {
                    "case_id": f"S{c:05d}",
                    "activity": str(rng.choice(activities)),
                    "timestamp": pd.to_datetime(t, unit="s", utc=True),
                    "lifecycle": "complete",
                    "resource": "System",
                }
            )
    df = pd.DataFrame(rows).sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    return df


def time_based_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    starts = df.groupby("case_id")["timestamp"].min().reset_index()
    starts = starts.sort_values("timestamp").reset_index(drop=True)
    n = len(starts)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_cases = set(starts.iloc[:n_train]["case_id"])
    val_cases = set(starts.iloc[n_train : n_train + n_val]["case_id"])
    test_cases = set(starts.iloc[n_train + n_val :]["case_id"])
    return train_cases, val_cases, test_cases


def build_prefix_dataset(df: pd.DataFrame, max_prefix_len=10, min_prefix_len=1):
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
        ts = (
            pd.to_datetime(g["timestamp"], utc=True).astype("int64").to_numpy() // 10**9
        )
        acts_ids = np.array(
            [act2id[a] for a in g["activity"].astype(str)], dtype=np.int64
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
        if T < 2:
            continue
        max_k = min(max_prefix_len, T - 1)
        for k in range(min_prefix_len, max_k + 1):
            samples.append(
                {
                    "case_id": cid,
                    "seq_acts": acts_ids[:k].tolist(),
                    "seq_feats": feats[:k].copy(),
                    "target": int(acts_ids[k]),
                }
            )
    if len(samples) > 0:
        all_feats = np.concatenate(
            [s["seq_feats"] for s in samples if s["seq_feats"].shape[0] > 0], axis=0
        )
        dt_mean, dt_std = all_feats[:, 0].mean(), all_feats[:, 0].std() + 1e-6
        ss_mean, ss_std = all_feats[:, 1].mean(), all_feats[:, 1].std() + 1e-6
        for s in samples:
            if s["seq_feats"].shape[0] > 0:
                s["seq_feats"][:, 0] = (s["seq_feats"][:, 0] - dt_mean) / dt_std
                s["seq_feats"][:, 1] = (s["seq_feats"][:, 1] - ss_mean) / ss_std
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


def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> float:
    if probs.size == 0 or y_true.size == 0:
        return 0.0
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        acc_bin = correct[m].mean()
        conf_bin = conf[m].mean()
        ece += (m.mean()) * abs(acc_bin - conf_bin)
    return float(ece)


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
    y_true = np.array(ys, dtype=np.int64)
    y_pred = np.array(preds_top1, dtype=np.int64)
    probs_concat = (
        np.concatenate(preds_probs, axis=0)
        if len(preds_probs) > 0
        else np.zeros((0, num_classes + 1), dtype=np.float32)
    )
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    try:
        f1 = float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        f1 = 0.0
    top3 = float(top3_correct / max(1, n_total))
    ece = (
        expected_calibration_error(probs_concat, y_true, n_bins=15)
        if probs_concat.shape[0] > 0
        else 0.0
    )
    return avg_loss, acc, f1, top3, ece, y_true, y_pred, probs_concat


def train_eval_one_setting(
    ds_name,
    df,
    batch_size=128,
    max_epochs=12,
    max_prefix_len=12,
    lr=1e-3,
    cap_cases=8000,
):
    try:
        starts = (
            df.groupby("case_id")["timestamp"]
            .min()
            .reset_index()
            .sort_values("timestamp")
        )
        if len(starts) > cap_cases:
            keep_cases = set(starts.iloc[:cap_cases]["case_id"])
            df = df[df["case_id"].isin(keep_cases)].copy()
    except Exception:
        pass
    train_cases, val_cases, test_cases = time_based_split(df, 0.70, 0.15)
    samples_all, act2id, id2act, pad_id = build_prefix_dataset(
        df, max_prefix_len=max_prefix_len
    )
    samples_train = [s for s in samples_all if s["case_id"] in train_cases]
    samples_val = [s for s in samples_all if s["case_id"] in val_cases]
    samples_test = [s for s in samples_all if s["case_id"] in test_cases]
    if len(samples_train) > 0:
        concat_feats = [
            s["seq_feats"] for s in samples_train if s["seq_feats"].shape[0] > 0
        ]
        if len(concat_feats) > 0:
            all_feats = np.concatenate(concat_feats, axis=0)
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
        f"[{ds_name}] samples train/val/test: {len(samples_train)}/{len(samples_val)}/{len(samples_test)}, vocab={len(act2id)}"
    )
    if len(samples_train) == 0 or len(act2id) < 2:
        print(f"[{ds_name}] Not enough data. Skipping.")
        return None
    ds_tr = PrefixDataset(
        samples_train, pad_id=pad_id, max_len=max_prefix_len, num_cont=5
    )
    ds_va = PrefixDataset(
        samples_val, pad_id=pad_id, max_len=max_prefix_len, num_cont=5
    )
    ds_te = PrefixDataset(
        samples_test, pad_id=pad_id, max_len=max_prefix_len, num_cont=5
    )
    dl_tr = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    dl_te = DataLoader(
        ds_te,
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
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_top3": [],
        "val_ece": [],
    }
    best_state = None
    best_val_top3 = -1.0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0
        run_loss = 0.0
        for batch in dl_tr:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            optimizer.zero_grad()
            logits = model(batch["acts"], batch["feats"], batch["mask"])
            loss = criterion(logits, batch["y"])
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * logits.size(0)
            total += logits.size(0)
        tr_loss = run_loss / max(1, total)
        val_loss, val_acc, val_f1, val_top3, val_ece, _, _, _ = evaluate(
            model, dl_va, criterion, device, len(act2id), pad_id
        )
        print(
            f"[{ds_name}][bs={batch_size}, lr={lr}] Epoch {epoch}: validation_loss = {val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} top3={val_top3:.4f} ece={val_ece:.4f}"
        )
        history["train_loss"].append((epoch, float(tr_loss)))
        history["val_loss"].append((epoch, float(val_loss)))
        history["val_acc"].append((epoch, float(val_acc)))
        history["val_f1"].append((epoch, float(val_f1)))
        history["val_top3"].append((epoch, float(val_top3)))
        history["val_ece"].append((epoch, float(val_ece)))
        if val_top3 > best_val_top3:
            best_val_top3 = val_top3
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    (
        train_loss,
        train_acc,
        train_f1,
        train_top3,
        train_ece,
        y_true_tr,
        y_pred_tr,
        probs_tr,
    ) = evaluate(model, dl_tr, criterion, device, len(act2id), pad_id)
    val_loss, val_acc, val_f1, val_top3, val_ece, y_true_va, y_pred_va, probs_va = (
        evaluate(model, dl_va, criterion, device, len(act2id), pad_id)
    )
    (
        test_loss,
        test_acc,
        test_f1,
        test_top3,
        test_ece,
        y_true_te,
        y_pred_te,
        probs_te,
    ) = evaluate(model, dl_te, criterion, device, len(act2id), pad_id)

    def plot_calibration(probs, y_true, title, out_path):
        if probs.size == 0:
            return
        conf = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        correct = (preds == y_true).astype(np.float32)
        bins = np.linspace(0.0, 1.0, 11)
        bin_ids = np.digitize(conf, bins) - 1
        accs, confs = [], []
        for i in range(10):
            m = bin_ids == i
            if m.sum() == 0:
                accs.append(0.0)
                confs.append((bins[i] + bins[i + 1]) / 2)
            else:
                accs.append(correct[m].mean())
                confs.append(conf[m].mean())
        plt.figure(figsize=(4, 4))
        plt.plot([0, 1], [0, 1], "k--", label="Perfect")
        plt.plot(confs, accs, marker="o", label="Model")
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    plot_calibration(
        probs_te,
        y_true_te,
        f"{ds_name} Reliability (Test)",
        os.path.join(working_dir, f"{ds_name}_reliability_test.png"),
    )
    results = {
        "settings": {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "max_prefix_len": max_prefix_len,
            "lr": lr,
            "vocab_size": len(act2id),
        },
        "losses": {"train": history["train_loss"], "val": history["val_loss"]},
        "metrics": {
            "train_final": {
                "loss": float(train_loss),
                "acc": float(train_acc),
                "macro_f1": float(train_f1),
                "top3": float(train_top3),
                "ece": float(train_ece),
            },
            "val_final": {
                "loss": float(val_loss),
                "acc": float(val_acc),
                "macro_f1": float(val_f1),
                "top3": float(val_top3),
                "ece": float(val_ece),
            },
            "test_final": {
                "loss": float(test_loss),
                "acc": float(test_acc),
                "macro_f1": float(test_f1),
                "top3": float(test_top3),
                "ece": float(test_ece),
            },
            "val_curves": {
                "acc": history["val_acc"],
                "macro_f1": history["val_f1"],
                "top3": history["val_top3"],
                "ece": history["val_ece"],
            },
        },
        "predictions": {
            "train": np.array(y_pred_tr, dtype=np.int64),
            "val": np.array(y_pred_va, dtype=np.int64),
            "test": np.array(y_pred_te, dtype=np.int64),
        },
        "ground_truth": {
            "train": np.array(y_true_tr, dtype=np.int64),
            "val": np.array(y_true_va, dtype=np.int64),
            "test": np.array(y_true_te, dtype=np.int64),
        },
        "probs": {"train": probs_tr, "val": probs_va, "test": probs_te},
        "act2id": act2id,
        "pad_idx": int(pad_id),
    }
    return results


def run_experiments():
    datasets = load_datasets()
    if len(datasets) == 0:
        print("[info] No XES datasets found. Using synthetic dataset as fallback.")
        datasets = {"SYNTH": make_synthetic_df()}
    batch_grid = [64, 128, 256]
    lr_grid = [5e-4, 1e-3]
    max_epochs = 12
    max_prefix_len = 12
    for ds_name, df in datasets.items():
        experiment_data.setdefault(
            ds_name,
            {
                "metrics": {"train": [], "val": [], "test": []},
                "losses": {"train": [], "val": []},
                "predictions": [],
                "ground_truth": [],
                "settings": {},
                "best": {},
            },
        )
        best = {"score": -1.0, "bs": None, "lr": None, "result": None}
        for bs in batch_grid:
            for lr in lr_grid:
                try:
                    res = train_eval_one_setting(
                        ds_name,
                        df,
                        batch_size=bs,
                        max_epochs=max_epochs,
                        max_prefix_len=max_prefix_len,
                        lr=lr,
                    )
                except RuntimeError as e:
                    print(
                        f"[warn][{ds_name}] Runtime error for batch_size={bs}, lr={lr}: {e}"
                    )
                    res = None
                key = f"bs{bs}_lr{lr}"
                if res is None:
                    continue
                experiment_data[ds_name]["losses"]["train"].extend(
                    res["losses"]["train"]
                )
                experiment_data[ds_name]["losses"]["val"].extend(res["losses"]["val"])
                experiment_data[ds_name]["metrics"]["train"].append(
                    ("final", res["metrics"]["train_final"])
                )
                experiment_data[ds_name]["metrics"]["val"].append(
                    ("final", res["metrics"]["val_final"])
                )
                experiment_data[ds_name]["metrics"]["test"].append(
                    ("final", res["metrics"]["test_final"])
                )
                experiment_data[ds_name]["settings"][key] = res["settings"]
                experiment_data[ds_name]["predictions"] = res["predictions"]["test"]
                experiment_data[ds_name]["ground_truth"] = res["ground_truth"]["test"]
                val_top3 = res["metrics"]["val_final"]["top3"]
                if val_top3 > best["score"]:
                    best = {"score": val_top3, "bs": bs, "lr": lr, "result": res}
        if best["result"] is not None:
            experiment_data[ds_name]["best"] = {
                "batch_size": best["bs"],
                "lr": best["lr"],
                "val_top3": float(best["score"]),
                "test_acc": float(best["result"]["metrics"]["test_final"]["acc"]),
                "test_top3": float(best["result"]["metrics"]["test_final"]["top3"]),
                "test_macro_f1": float(
                    best["result"]["metrics"]["test_final"]["macro_f1"]
                ),
                "test_ece": float(best["result"]["metrics"]["test_final"]["ece"]),
            }
            print(
                f"[{ds_name}] Best setting: bs={best['bs']} lr={best['lr']} | val_top3={best['score']:.4f} | test_acc={best['result']['metrics']['test_final']['acc']:.4f} test_top3={best['result']['metrics']['test_final']['top3']:.4f} ece={best['result']['metrics']['test_final']['ece']:.4f}"
            )
        else:
            print(f"[{ds_name}] No successful runs.")
    np.save(
        os.path.join(working_dir, "experiment_data.npy"),
        experiment_data,
        allow_pickle=True,
    )


# Train and save
run_experiments()

# Load and Plot
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for ds_name, ds_blob in experiment_data.items():
    # 1) Training/Validation Loss Curves
    try:
        plt.figure()
        tr = ds_blob.get("losses", {}).get("train", [])
        va = ds_blob.get("losses", {}).get("val", [])
        if len(tr) > 0 or len(va) > 0:
            if len(tr) > 0:
                epochs_tr = [e for e, v in tr]
                vals_tr = [v for e, v in tr]
                plt.plot(epochs_tr, vals_tr, label="Train Loss")
            if len(va) > 0:
                epochs_va = [e for e, v in va]
                vals_va = [v for e, v in va]
                plt.plot(epochs_va, vals_va, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{ds_name} Loss Curves (Next-Activity)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_loss_curves_next_activity.png")
            )
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {ds_name}: {e}")
        plt.close()

    # 2) Confusion Matrix (Test)
    try:
        y_true = np.array(ds_blob.get("ground_truth", []))
        y_pred = np.array(ds_blob.get("predictions", []))
        if isinstance(y_true, dict):
            y_true = y_true.get("test", np.array([]))
        if isinstance(y_pred, dict):
            y_pred = y_pred.get("test", np.array([]))
        if y_true.size > 0 and y_pred.size > 0:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{ds_name} Confusion Matrix (Next-Activity, Test)")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, f"{ds_name}_confusion_matrix_next_activity_test.png"
                )
            )
            plt.close()
        else:
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        plt.close()

    # 3) Macro Precision-Recall Curve (Test)
    try:
        probs = ds_blob.get("probs", {}).get("test", None)
        y_true = ds_blob.get("ground_truth", {}).get("test", None)
        if (
            probs is not None
            and y_true is not None
            and len(probs) == len(y_true)
            and len(y_true) > 0
        ):
            n_classes = probs.shape[1]
            # Ignore padding class 0 if present by masking columns that are never true
            classes = np.unique(y_true)
            y_true_mat = np.zeros((len(y_true), n_classes), dtype=int)
            for i, c in enumerate(y_true):
                if c >= 0 and c < n_classes:
                    y_true_mat[i, c] = 1
            pr_areas = []
            precisions_macro = []
            recalls_macro = []
            # Compute macro by averaging interpolated precision across classes present in truth
            valid_classes = [c for c in range(n_classes) if y_true_mat[:, c].sum() > 0]
            grid = np.linspace(0, 1, 101)
            prec_accum = np.zeros_like(grid)
            for c in valid_classes:
                prec, rec, _ = precision_recall_curve(y_true_mat[:, c], probs[:, c])
                # interpolate precision as function of recall
                prec_interp = np.interp(
                    grid, rec[::-1], prec[::-1], left=prec[0], right=prec[-1]
                )
                prec_accum += prec_interp
                try:
                    ap = average_precision_score(y_true_mat[:, c], probs[:, c])
                    pr_areas.append(ap)
                except Exception:
                    pass
            if len(valid_classes) > 0:
                prec_macro = prec_accum / len(valid_classes)
                plt.figure()
                plt.plot(
                    grid, prec_macro, label=f"Macro-PR (AP~{np.mean(pr_areas):.3f})"
                )
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"{ds_name} Macro PR Curve (Next-Activity, Test)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        working_dir, f"{ds_name}_macro_pr_next_activity_test.png"
                    )
                )
                plt.close()
            else:
                plt.close()
        else:
            plt.close()
    except Exception as e:
        print(f"Error creating macro PR curve for {ds_name}: {e}")
        plt.close()

# Print summary BPM metrics
for ds_name, ds_blob in experiment_data.items():
    best = ds_blob.get("best", {})
    if best:
        print(
            f"[RESULT] {ds_name} | Best bs={best.get('batch_size')} lr={best.get('lr')} | "
            f"Test Acc={best.get('test_acc'):.4f} | Test Top-3={best.get('test_top3'):.4f} | "
            f"Test Macro-F1={best.get('test_macro_f1'):.4f} | Test ECE={best.get('test_ece'):.4f}"
        )
    else:
        # try to print any available final test from first metric entry
        tests = ds_blob.get("metrics", {}).get("test", [])
        if len(tests) > 0 and isinstance(tests[0], tuple):
            m = tests[0][1]
            print(
                f"[RESULT] {ds_name} | Test Acc={m.get('acc', float('nan')):.4f} | Top-3={m.get('top3', float('nan')):.4f} | Macro-F1={m.get('macro_f1', float('nan')):.4f} | ECE={m.get('ece', float('nan')):.4f}"
            )
