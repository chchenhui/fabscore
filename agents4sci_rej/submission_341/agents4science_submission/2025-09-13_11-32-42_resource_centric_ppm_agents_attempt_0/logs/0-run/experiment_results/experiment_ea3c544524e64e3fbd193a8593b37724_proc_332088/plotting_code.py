import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

os.makedirs(working_dir, exist_ok=True)

import warnings

warnings.filterwarnings("ignore")
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
from datetime import timedelta
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experiment data container
experiment_data = {
    "DEFAULT": {
        "name": "",
        "task": "next-activity",
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "top3_history": {"train": [], "val": []},
    }
}


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
    raise FileNotFoundError("No .xes found")


def xes_to_df(xes_path: Path) -> pd.DataFrame:
    try:
        from pm4py.objects.log.importer.xes import importer as xes_importer
    except Exception as e:
        raise ImportError("pm4py is required. pip install pm4py") from e
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
    df = df.sort_values(["timestamp", "case_id"]).reset_index(drop=True)
    return df


def load_default_dataset() -> Tuple[str, pd.DataFrame]:
    datasets = {}
    try:
        d = _resolve_data_dir()
        files = list(d.glob("*.xes")) + list(d.glob("*.xes.gz"))
        pref = [
            "BPI_Challenge_2017",
            "BPI2017",
            "2017",
            "BPI_Challenge_2012",
            "BPI2012",
            "2012",
            "Road_Traffic_Fine_Management_Process",
            "Traffic",
            "Fine",
            "Road",
        ]
        chosen = None
        for p in pref:
            for f in files:
                if p in f.name:
                    chosen = f
                    break
            if chosen is not None:
                break
        if chosen is None and files:
            chosen = files[0]
        if chosen is not None:
            df = xes_to_df(chosen)
            name = (
                "BPI2017"
                if "2017" in chosen.name
                else ("BPI2012" if "2012" in chosen.name else "ROAD")
            )
            return name, df
    except Exception as e:
        print(f"[warn] Data discovery failed: {e}")
    print("[data] Generating synthetic log")
    rng = np.random.RandomState(42)
    activities = ["A", "B", "C", "D", "E", "F"]
    resources = ["R1", "R2", "R3"]
    rows = []
    start = pd.Timestamp("2020-01-01", tz="UTC")
    n_cases = 200
    for c in range(n_cases):
        case_id = f"C{c:04d}"
        t = start + pd.Timedelta(days=int(c / 5))
        length = rng.randint(4, 9)
        seq = ["A"]
        for i in range(length - 1):
            if seq[-1] == "A":
                nxt = "B" if rng.rand() < 0.6 else "C"
            elif seq[-1] in ["B", "C"]:
                nxt = "D" if rng.rand() < 0.5 else "E"
            elif seq[-1] in ["D", "E"]:
                nxt = "F" if rng.rand() < 0.7 else rng.choice(["B", "C"])
            else:
                nxt = rng.choice(activities)
            seq.append(nxt)
        for act in seq:
            rows.append(
                {
                    "case_id": case_id,
                    "activity": act,
                    "lifecycle": "complete",
                    "timestamp": t,
                    "resource": rng.choice(resources),
                }
            )
            t += pd.Timedelta(minutes=int(rng.exponential(120)))
    df = pd.DataFrame(rows).sort_values(["timestamp", "case_id"]).reset_index(drop=True)
    return "SYNTH", df


def build_prefix_samples(
    df: pd.DataFrame, max_len: int = 10
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    g = df.sort_values("timestamp").groupby("case_id")
    traces = []
    for cid, grp in g:
        acts = grp["activity"].tolist()
        times = grp["timestamp"].tolist()
        res = (
            grp["resource"].tolist()
            if "resource" in grp.columns
            else ["System"] * len(acts)
        )
        traces.append((cid, acts, times, res))
    act_set = sorted({a for _, acts, _, _ in traces for a in acts})
    act2ix = {a: i + 1 for i, a in enumerate(act_set)}  # 0 PAD
    samples = []
    for cid, acts, times, res in traces:
        if len(acts) < 2:
            continue
        for i in range(1, min(len(acts), max_len + 1)):
            if i >= len(acts):
                break
            prefix_acts = acts[:i]
            next_act = acts[i]
            prefix_times = times[:i]
            start_t = prefix_times[0]
            last_t = prefix_times[-1]
            delta_start = (last_t - start_t).total_seconds() / 3600.0
            delta_last = (
                ((prefix_times[-1] - prefix_times[-2]).total_seconds() / 3600.0)
                if len(prefix_times) >= 2
                else 0.0
            )
            hour = last_t.hour
            weekday = last_t.weekday()
            working = 1.0 if (weekday < 5 and 8 <= hour < 18) else 0.0
            samples.append(
                {
                    "case_id": cid,
                    "seq": [act2ix[a] for a in prefix_acts][-max_len:],
                    "len": min(len(prefix_acts), max_len),
                    "aux": np.array(
                        [delta_start, delta_last, hour, weekday, working],
                        dtype=np.float32,
                    ),
                    "y": act2ix[next_act],
                }
            )
    data = pd.DataFrame(samples)
    return data, act2ix


def time_based_split(
    df: pd.DataFrame, train_frac=0.7, val_frac=0.1
) -> Tuple[List[str], List[str], List[str]]:
    starts = df.groupby("case_id")["timestamp"].min().sort_values()
    n = len(starts)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (
        starts.index[:train_end].tolist(),
        starts.index[train_end:val_end].tolist(),
        starts.index[val_end:].tolist(),
    )


class PrefixDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cases: List[str],
        scaler: StandardScaler = None,
        fit_scaler=False,
        max_len=10,
    ):
        self.samples = df[df["case_id"].isin(cases)].reset_index(drop=True)
        self.max_len = max_len
        aux = np.stack(self.samples["aux"].values).astype(np.float32)
        if scaler is None:
            scaler = StandardScaler()
        if fit_scaler:
            scaler.fit(aux)
        self.scaler = scaler
        aux_norm = scaler.transform(aux).astype(np.float32)
        self.samples["aux_norm"] = list(aux_norm)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        seq = row["seq"]
        L = min(len(seq), self.max_len)
        padded = np.zeros(self.max_len, dtype=np.int64)
        padded[-L:] = np.array(seq[-L:], dtype=np.int64)
        return {
            "seq": torch.tensor(padded, dtype=torch.long),
            "len": torch.tensor(L, dtype=torch.long),
            "aux": torch.tensor(np.array(row["aux_norm"], dtype=np.float32)),
            "y": torch.tensor(row["y"], dtype=torch.long),
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


class NextActLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 32,
        hidden: int = 64,
        aux_dim: int = 5,
        num_classes: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden + aux_dim, num_classes)

    def forward(self, seq, lengths, aux):
        emb = self.emb(seq)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]
        x = self.dropout(torch.cat([h_last, aux], dim=1))
        return self.fc(x)


def eval_epoch(model, loader):
    model.eval()
    all_y, all_pred, all_logits = [], [], []
    loss_sum, n = 0.0, 0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = model(batch["seq"], batch["len"], batch["aux"])
            loss = ce(logits, batch["y"])
            loss_sum += loss.item() * batch["y"].size(0)
            n += batch["y"].size(0)
            preds = torch.argmax(logits, dim=1)
            all_y.append(batch["y"].cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    y_true = np.concatenate(all_y) if all_y else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    logits_all = (
        np.concatenate(all_logits)
        if all_logits
        else np.zeros((0, model.fc.out_features))
    )
    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    macro_f1 = (
        f1_score(y_true, y_pred, average="macro", zero_division=0)
        if len(y_true) > 0
        else 0.0
    )
    if len(y_true) > 0 and logits_all.shape[0] > 0:
        top3 = [
            int(y_true[i] in np.argsort(-logits_all[i])[:3])
            for i in range(logits_all.shape[0])
        ]
        top3_acc = float(np.mean(top3)) if top3 else 0.0
    else:
        top3_acc = 0.0
    return {
        "loss": loss_sum / max(n, 1),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "top3_acc": top3_acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "logits": logits_all,
    }


def run_experiment():
    ds_name, df = load_default_dataset()
    print(
        f"[data] Dataset: {ds_name}, events={len(df)}, cases={df['case_id'].nunique()}"
    )
    max_len = 10
    prefixes, act2ix = build_prefix_samples(df, max_len=max_len)
    print(f"[data] Prefix samples: {len(prefixes)}, vocab={len(act2ix)}")
    if len(prefixes) == 0 or len(act2ix) < 2:
        print("[warn] Not enough data to train.")
        return ds_name
    train_cases, val_cases, test_cases = time_based_split(df)
    scaler = StandardScaler()
    train_ds = PrefixDataset(
        prefixes, train_cases, scaler=scaler, fit_scaler=True, max_len=max_len
    )
    val_ds = PrefixDataset(
        prefixes, val_cases, scaler=scaler, fit_scaler=False, max_len=max_len
    )
    test_ds = PrefixDataset(
        prefixes, test_cases, scaler=scaler, fit_scaler=False, max_len=max_len
    )
    print(f"[split] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn
    )
    vocab_size = len(act2ix) + 1
    num_classes = len(act2ix) + 1
    model = NextActLSTM(
        vocab_size=vocab_size,
        emb_dim=32,
        hidden=64,
        aux_dim=5,
        num_classes=num_classes,
        dropout=0.2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss()
    epochs = 8
    exp_key = "DEFAULT"
    experiment_data[exp_key]["name"] = ds_name
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            logits = model(batch["seq"], batch["len"], batch["aux"])
            loss = ce(logits, batch["y"])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch["y"].size(0)
            n_seen += batch["y"].size(0)
        train_loss /= max(n_seen, 1)
        train_metrics = eval_epoch(model, train_loader)
        val_metrics = eval_epoch(model, val_loader)
        print(f"Epoch {epoch}: validation_loss = {val_metrics['loss']:.4f}")
        experiment_data[exp_key]["epochs"].append(epoch)
        experiment_data[exp_key]["losses"]["train"].append(
            {"epoch": epoch, "loss": train_loss}
        )
        experiment_data[exp_key]["losses"]["val"].append(
            {"epoch": epoch, "loss": val_metrics["loss"]}
        )
        experiment_data[exp_key]["metrics"]["train"].append(
            {
                "epoch": epoch,
                "accuracy": train_metrics["accuracy"],
                "macro_f1": train_metrics["macro_f1"],
                "top3_acc": train_metrics["top3_acc"],
            }
        )
        experiment_data[exp_key]["metrics"]["val"].append(
            {
                "epoch": epoch,
                "accuracy": val_metrics["accuracy"],
                "macro_f1": val_metrics["macro_f1"],
                "top3_acc": val_metrics["top3_acc"],
            }
        )
        experiment_data[exp_key]["top3_history"]["train"].append(
            train_metrics["top3_acc"]
        )
        experiment_data[exp_key]["top3_history"]["val"].append(val_metrics["top3_acc"])
    test_metrics = eval_epoch(model, test_loader)
    experiment_data[exp_key]["metrics"]["test"].append(
        {
            "epoch": epochs,
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "top3_acc": test_metrics["top3_acc"],
        }
    )
    experiment_data[exp_key]["predictions"] = test_metrics["y_pred"].astype(int)
    experiment_data[exp_key]["ground_truth"] = test_metrics["y_true"].astype(int)
    last_train = experiment_data[exp_key]["metrics"]["train"][-1]
    last_val = experiment_data[exp_key]["metrics"]["val"][-1]
    print(
        f"[results][{ds_name}] Train: Acc={last_train['accuracy']:.4f} MacroF1={last_train['macro_f1']:.4f} Top-3={last_train['top3_acc']:.4f}"
    )
    print(
        f"[results][{ds_name}] Val:   Acc={last_val['accuracy']:.4f} MacroF1={last_val['macro_f1']:.4f} Top-3={last_val['top3_acc']:.4f}"
    )
    print(
        f"[results][{ds_name}] Test:  Acc={test_metrics['accuracy']:.4f} MacroF1={test_metrics['macro_f1']:.4f} Top-3={test_metrics['top3_acc']:.4f}"
    )
    # Save curves immediately
    try:
        epochs_arr = experiment_data[exp_key]["epochs"]
        tr_losses = [x["loss"] for x in experiment_data[exp_key]["losses"]["train"]]
        vl_losses = [x["loss"] for x in experiment_data[exp_key]["losses"]["val"]]
        plt.figure()
        plt.plot(epochs_arr, tr_losses, label="train_loss")
        plt.plot(epochs_arr, vl_losses, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Loss Curves - {ds_name} (Next-Activity)")
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_nextact_loss_curves.png"), dpi=150
        )
        plt.close()
        tr_top3 = experiment_data[exp_key]["top3_history"]["train"]
        vl_top3 = experiment_data[exp_key]["top3_history"]["val"]
        plt.figure()
        plt.plot(epochs_arr, tr_top3, label="train_top3")
        plt.plot(epochs_arr, vl_top3, label="val_top3")
        plt.xlabel("Epoch")
        plt.ylabel("Top-3 Accuracy")
        plt.legend()
        plt.title(f"Top-3 Accuracy Curves - {ds_name} (Next-Activity)")
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_nextact_top3_curves.png"), dpi=150
        )
        plt.close()
    except Exception as e:
        print(f"[warn] Plotting failed: {e}")
        plt.close()
    # Save experiment data
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    np.savez_compressed(
        os.path.join(working_dir, f"preds_{ds_name}.npz"),
        y_true=experiment_data[exp_key]["ground_truth"],
        y_pred=experiment_data[exp_key]["predictions"],
    )
    print(
        f"Top-3 Next-Activity Accuracy (TEST) [{ds_name}]: {test_metrics['top3_acc']:.4f}"
    )
    # Return final metrics for printing
    return ds_name, test_metrics


# Execute training + evaluation
ds_name, test_metrics = run_experiment()

# Load experiment_data and create standard visualizations (only from available data)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Plot 1: Training/Validation Loss Curves
try:
    exp_key = "DEFAULT"
    ds = experiment_data.get(exp_key, {})
    name = ds.get("name", "DATASET")
    epochs_arr = ds.get("epochs", [])
    if (
        epochs_arr
        and ds.get("losses", {}).get("train")
        and ds.get("losses", {}).get("val")
    ):
        tr_losses = [x["loss"] for x in ds["losses"]["train"]]
        vl_losses = [x["loss"] for x in ds["losses"]["val"]]
        plt.figure()
        plt.plot(epochs_arr, tr_losses, label="Train Loss")
        plt.plot(epochs_arr, vl_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"{name} - Next-Activity: Loss Curves")
        plt.suptitle("Left: Train, Right: N/A (single view)")
        plt.savefig(
            os.path.join(working_dir, f"{name}_nextact_loss_curves_from_expdata.png"),
            dpi=150,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Top-3 Accuracy Curves
try:
    exp_key = "DEFAULT"
    ds = experiment_data.get(exp_key, {})
    name = ds.get("name", "DATASET")
    epochs_arr = ds.get("epochs", [])
    tr_top3 = ds.get("top3_history", {}).get("train", [])
    vl_top3 = ds.get("top3_history", {}).get("val", [])
    if epochs_arr and tr_top3 and vl_top3:
        plt.figure()
        plt.plot(epochs_arr, tr_top3, label="Train Top-3")
        plt.plot(epochs_arr, vl_top3, label="Val Top-3")
        plt.xlabel("Epoch")
        plt.ylabel("Top-3 Accuracy")
        plt.legend()
        plt.title(f"{name} - Next-Activity: Top-3 Accuracy Curves")
        plt.suptitle("Left: Train, Right: Validation")
        plt.savefig(
            os.path.join(working_dir, f"{name}_nextact_top3_curves_from_expdata.png"),
            dpi=150,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Test Confusion Matrix
try:
    exp_key = "DEFAULT"
    ds = experiment_data.get(exp_key, {})
    name = ds.get("name", "DATASET")
    y_true = np.array(ds.get("ground_truth", []))
    y_pred = np.array(ds.get("predictions", []))
    if y_true.size > 0 and y_pred.size > 0:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title(f"{name} - Next-Activity: Confusion Matrix (Test)")
        plt.suptitle("Left: Ground Truth, Right: Predicted (Indices)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        ticks = np.arange(cm.shape[0])
        if len(ticks) <= 30:
            plt.xticks(ticks)
            plt.yticks(ticks)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{name}_nextact_confusion_matrix_test.png"),
            dpi=150,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Print evaluation metrics prominently
try:
    print(f"TEST Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"TEST Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"TEST Top-3 Accuracy: {test_metrics['top3_acc']:.4f}")
except Exception as e:
    print(f"Error printing metrics: {e}")
