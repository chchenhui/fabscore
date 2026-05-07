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
from sklearn.preprocessing import label_binarize
import random
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def load_xes_folder(data_dir="data"):
    datasets = {}
    try:
        import pm4py
    except Exception as e:
        print(f"pm4py not available: {e}")
        return datasets
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        return datasets
    for fn in os.listdir(data_dir):
        if fn.lower().endswith(".xes") or fn.lower().endswith(".xes.gz"):
            path = os.path.join(data_dir, fn)
            try:
                log = pm4py.read_xes(path)
                df = pm4py.convert_to_dataframe(log)
                cols = df.columns
                case_col = (
                    "case:concept:name"
                    if "case:concept:name" in cols
                    else ("case" if "case" in cols else None)
                )
                act_col = (
                    "concept:name"
                    if "concept:name" in cols
                    else ("activity" if "activity" in cols else None)
                )
                ts_col = (
                    "time:timestamp"
                    if "time:timestamp" in cols
                    else ("timestamp" if "timestamp" in cols else None)
                )
                life_col = (
                    "lifecycle:transition"
                    if "lifecycle:transition" in cols
                    else ("lifecycle" if "lifecycle" in cols else None)
                )
                if case_col is None or act_col is None or ts_col is None:
                    print(f"Missing required columns in {fn}, skipping.")
                    continue
                out = pd.DataFrame(
                    {
                        "case_id": df[case_col].astype(str).values,
                        "activity": df[act_col].astype(str).values,
                        "timestamp": pd.to_datetime(df[ts_col], utc=True),
                    }
                )
                if life_col is not None:
                    out["lifecycle"] = df[life_col].astype(str).values
                name = os.path.splitext(fn)[0]
                print(
                    f"Loaded {name}: {len(out)} events, {out['case_id'].nunique()} cases"
                )
                datasets[name] = out
            except Exception as e:
                print(f"Failed to load {fn}: {e}")
    return datasets


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
    df = df.copy()
    if "lifecycle" in df.columns:
        mask = df["lifecycle"].astype(str).str.lower().eq("complete")
        if mask.any():
            df = df[mask]
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
        g_ts = pd.to_datetime(g["timestamp"], utc=True)
        ts = (g_ts.astype("int64") // 10**9).to_numpy(np.int64)
        acts_ids = np.array(
            [act2id[a] for a in g["activity"].astype(str)], dtype=np.int64
        )
        hours = (g_ts.dt.hour.to_numpy(float) / 23.0).astype(np.float32)
        weekdays = (g_ts.dt.weekday.to_numpy(float) / 6.0).astype(np.float32)
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
                    "prefix_len": k,
                }
            )
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
        feats_pad = np.vstack(
            [np.zeros((pad, self.num_cont), np.float32), feats.astype(np.float32)]
        )
        mask = np.array([0] * pad + [1] * L, np.float32)
        return {
            "acts": torch.tensor(seq_pad).long(),
            "feats": torch.tensor(feats_pad).float(),
            "mask": torch.tensor(mask).float(),
            "y": torch.tensor(s["target"]).long(),
            "prefix_len": L,
        }


def collate_fn(batch):
    return {
        k: (
            torch.stack([b[k] for b in batch], 0)
            if isinstance(batch[0][k], torch.Tensor)
            else [b[k] for b in batch]
        )
        for k in batch[0].keys()
    }


class LSTMBaseline(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, cont_dim=5, hidden=128, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim + cont_dim, hidden_size=hidden, batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden, vocab_size + 1)

    def forward(self, acts, feats, mask):
        x = self.emb(acts)
        x = torch.cat([x, feats], -1)
        out, (h, c) = self.lstm(x)
        h = self.dropout(h[-1])
        return self.fc(h)


def evaluate(model, loader, criterion, device, num_classes, pad_idx):
    model.eval()
    total_loss = 0.0
    ys = []
    yhat = []
    probs_list = []
    n = 0
    top3_correct = 0
    pref_lens = []
    top3_flags = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["acts"], batch["feats"], batch["mask"])
            loss = criterion(logits, batch["y"])
            total_loss += loss.item() * logits.size(0)
            probs = torch.softmax(logits, dim=1)
            top1 = torch.argmax(probs, dim=1)
            k_val = min(3, probs.size(1))
            _, topk = torch.topk(probs, k=k_val, dim=1)
            y = batch["y"]
            ys.extend(y.detach().cpu().tolist())
            yhat.extend(top1.detach().cpu().tolist())
            probs_list.append(probs.detach().cpu().numpy())
            for i in range(y.size(0)):
                flag = int(y[i].item() in topk[i].detach().cpu().tolist())
                top3_correct += flag
                top3_flags.append(flag)
                pref_lens.append(int(batch["prefix_len"][i]))
            n += y.size(0)
    avg_loss = total_loss / max(1, n)
    y_true = np.array(ys)
    y_pred = np.array(yhat)
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    try:
        f1 = float(f1_score(y_true, y_pred, average="macro"))
    except:
        f1 = 0.0
    top3 = float(top3_correct / max(1, n))
    probs_concat = (
        np.concatenate(probs_list, 0)
        if len(probs_list) > 0
        else np.zeros((0, num_classes + 1))
    )
    return (
        avg_loss,
        acc,
        f1,
        top3,
        y_true,
        y_pred,
        probs_concat,
        np.array(pref_lens),
        np.array(top3_flags),
    )


def train_on_dataset(
    name, df, max_epochs=10, batch_size=128, max_prefix_len=10, lr=1e-3
):
    print(f"\n=== Dataset: {name} ===")
    train_cases, val_cases, test_cases = time_based_split(df, 0.7, 0.15)
    samples_all, act2id, id2act, pad_id = build_prefix_dataset(
        df, max_prefix_len=max_prefix_len
    )
    s_tr = [s for s in samples_all if s["case_id"] in train_cases]
    s_va = [s for s in samples_all if s["case_id"] in val_cases]
    s_te = [s for s in samples_all if s["case_id"] in test_cases]
    if len(s_tr) > 0:
        feats = np.concatenate(
            [s["seq_feats"] for s in s_tr if len(s["seq_feats"]) > 0], 0
        )
        dt_m, dt_s = feats[:, 0].mean(), feats[:, 0].std() + 1e-6
        ss_m, ss_s = feats[:, 1].mean(), feats[:, 1].std() + 1e-6
        for arr in (s_tr, s_va, s_te):
            for s in arr:
                if s["seq_feats"].shape[0] > 0:
                    s["seq_feats"][:, 0] = (s["seq_feats"][:, 0] - dt_m) / dt_s
                    s["seq_feats"][:, 1] = (s["seq_feats"][:, 1] - ss_m) / ss_s
    print(
        f"Samples train/val/test: {len(s_tr)}/{len(s_va)}/{len(s_te)}; vocab={len(act2id)}"
    )
    if len(s_tr) == 0 or len(act2id) < 2:
        print("Insufficient data; skipping.")
        return None
    dl_tr = DataLoader(
        PrefixDataset(s_tr, pad_id, max_prefix_len, 5),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dl_va = DataLoader(
        PrefixDataset(s_va, pad_id, max_prefix_len, 5),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    dl_te = DataLoader(
        PrefixDataset(s_te, pad_id, max_prefix_len, 5),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    model = LSTMBaseline(
        vocab_size=len(act2id), emb_dim=64, cont_dim=5, hidden=128, pad_idx=pad_id
    ).to(device)
    crit = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_top3 = -1.0
    best_state = None
    hist = {"train_loss": [], "val_loss": [], "val_top3": []}
    for ep in range(1, max_epochs + 1):
        model.train()
        tot = 0
        run = 0.0
        for batch in dl_tr:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            opt.zero_grad()
            logits = model(batch["acts"], batch["feats"], batch["mask"])
            loss = crit(logits, batch["y"])
            loss.backward()
            opt.step()
            run += loss.item() * logits.size(0)
            tot += logits.size(0)
        tr_loss = run / max(1, tot)
        va_loss, va_acc, va_f1, va_top3, *_ = evaluate(
            model, dl_va, crit, device, len(act2id), pad_id
        )
        print(
            f"Epoch {ep}: val_loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} top3={va_top3:.4f}"
        )
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["val_top3"].append(va_top3)
        if va_top3 > best_top3:
            best_top3 = va_top3
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    tr_loss, tr_acc, tr_f1, tr_top3, *_ = evaluate(
        model, dl_tr, crit, device, len(act2id), pad_id
    )
    te_loss, te_acc, te_f1, te_top3, y_true, y_pred, probs, pref_lens, top3_flags = (
        evaluate(model, dl_te, crit, device, len(act2id), pad_id)
    )
    print(
        f"[{name}] Test: loss={te_loss:.4f} acc={te_acc:.4f} f1={te_f1:.4f} top3={te_top3:.4f}"
    )
    exp = {
        "metrics": {
            "train": [
                (
                    "final",
                    {
                        "loss": tr_loss,
                        "acc": tr_acc,
                        "macro_f1": tr_f1,
                        "top3": tr_top3,
                    },
                )
            ],
            "val": [],
            "test": [
                (
                    "final",
                    {
                        "loss": te_loss,
                        "acc": te_acc,
                        "macro_f1": te_f1,
                        "top3": te_top3,
                    },
                )
            ],
        },
        "losses": {
            "train": list(enumerate(hist["train_loss"], start=1)),
            "val": list(enumerate(hist["val_loss"], start=1)),
        },
        "predictions": y_pred.tolist(),
        "ground_truth": y_true.tolist(),
        "epochs": list(range(1, len(hist["train_loss"]) + 1)),
        "probs": probs,
        "prefix_lens": pref_lens.tolist(),
        "top3_flags": top3_flags.tolist(),
        "act2id": act2id,
    }
    # Per-dataset plots
    try:
        plt.figure()
        plt.plot(hist["train_loss"], label="train")
        plt.plot(hist["val_loss"], label="val")
        plt.legend()
        plt.title(f"Loss Curves - {name}\nNext-activity")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {name}: {e}")
        plt.close()
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, aspect="auto", cmap="Blues")
        plt.colorbar()
        plt.title(f"Confusion Matrix (Test) - {name}\nNext-activity")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{name}_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {name}: {e}")
        plt.close()
    try:
        d = defaultdict(list)
        for L, flag in zip(pref_lens, top3_flags):
            d[int(L)].append(int(flag))
        if len(d) > 0:
            xs = sorted(d.keys())
            ys = [float(np.mean(d[k])) for k in xs]
            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.title(f"Top-3 Accuracy vs Prefix Length - {name}\nNext-activity")
            plt.xlabel("Prefix Length")
            plt.ylabel("Top-3 Accuracy")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{name}_top3_vs_prefixlen.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating Top-3 vs prefix length for {name}: {e}")
        plt.close()
    try:
        probs_arr = np.array(probs)
        if probs_arr.size > 0 and len(y_true) > 1:
            classes = sorted(set(y_true))
            Y = label_binarize(np.array(y_true), classes=range(probs_arr.shape[1]))
            present = classes
            if len(present) > 1:
                grid = np.linspace(0, 1, 101)
                curves = []
                aps = []
                for c in present:
                    p, r, _ = precision_recall_curve(Y[:, c], probs_arr[:, c])
                    curves.append(np.interp(grid, r[::-1], p[::-1]))
                    aps.append(average_precision_score(Y[:, c], probs_arr[:, c]))
                macro_p = np.mean(np.stack(curves, 0), 0)
                plt.figure()
                plt.plot(grid, macro_p, label=f"mAP={np.mean(aps):.3f}")
                plt.title(f"Macro Precision-Recall (Test) - {name}\nNext-activity")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(working_dir, f"{name}_macro_pr.png"))
                plt.close()
    except Exception as e:
        print(f"Error creating PR curve for {name}: {e}")
        plt.close()
    return name, exp


def main():
    datasets = load_xes_folder(data_dir=os.path.join(os.getcwd(), "data"))
    experiment_data = {}
    for name, df in datasets.items():
        try:
            starts = (
                df.sort_values("timestamp")
                .groupby("case_id")["timestamp"]
                .min()
                .reset_index()
            )
            if len(starts) > 5000:
                keep = set(starts.iloc[:5000]["case_id"])
                df = df[df["case_id"].isin(keep)].copy()
        except:
            pass
        res = train_on_dataset(
            name, df, max_epochs=10, batch_size=128, max_prefix_len=10, lr=1e-3
        )
        if res is not None:
            k, exp = res
            experiment_data[k] = exp
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    # Print evaluation metrics
    for k, v in experiment_data.items():
        tm = dict(v["metrics"]["test"][0][1])
        print(
            f"{k} | Test acc={tm['acc']:.4f} macro_f1={tm['macro_f1']:.4f} top3={tm['top3']:.4f} loss={tm['loss']:.4f}"
        )
    # Reload for plotting-only stage and cross-dataset comparisons
    try:
        experiment_data_loaded = np.load(
            os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
        ).item()
    except Exception as e:
        print(f"Error loading experiment data: {e}")
        experiment_data_loaded = {}
    # Cross-dataset comparison plots
    try:
        if len(experiment_data_loaded) > 0:
            names = []
            accs = []
            f1s = []
            t3 = []
            for n, ed in experiment_data_loaded.items():
                m = ed.get("metrics", {}).get("test", [])
                if len(m) > 0:
                    md = m[0][1]
                    names.append(n)
                    accs.append(md.get("acc", 0))
                    f1s.append(md.get("macro_f1", 0))
                    t3.append(md.get("top3", 0))
            if len(names) > 0:
                x = np.arange(len(names))
                w = 0.25
                plt.figure(figsize=(8, 4))
                plt.bar(x - w, accs, width=w, label="Acc")
                plt.bar(x, f1s, width=w, label="Macro-F1")
                plt.bar(x + w, t3, width=w, label="Top-3")
                plt.xticks(x, names, rotation=45, ha="right")
                plt.title("Cross-dataset Test Metrics\nNext-activity")
                plt.ylabel("Score")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(working_dir, "comparison_test_metrics.png"))
                plt.close()
    except Exception as e:
        print(f"Error creating plot1: {e}")
        plt.close()
    try:
        # Overlaid Top-3 vs Prefix Length across datasets (limit to at most 5 datasets)
        sel = list(experiment_data_loaded.keys())[:5]
        if len(sel) > 0:
            plt.figure()
            for n in sel:
                ed = experiment_data_loaded[n]
                pref = ed.get("prefix_lens", [])
                flags = ed.get("top3_flags", [])
                if len(pref) > 0 and len(flags) > 0:
                    d = defaultdict(list)
                    for L, f in zip(pref, flags):
                        d[int(L)].append(int(f))
                    xs = sorted(d.keys())
                    ys = [float(np.mean(d[k])) for k in xs]
                    if len(xs) > 0:
                        plt.plot(xs, ys, marker="o", label=n)
            plt.title("Top-3 vs Prefix Length (Test)\nNext-activity")
            plt.xlabel("Prefix Length")
            plt.ylabel("Top-3 Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "comparison_top3_vs_prefixlen.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating plot2: {e}")
        plt.close()
    # Dataset-specific re-plots from saved data to ensure compliance
    for name, ed in experiment_data_loaded.items():
        try:
            tl = [y for (_, y) in ed.get("losses", {}).get("train", [])]
            vl = [y for (_, y) in ed.get("losses", {}).get("val", [])]
            plt.figure()
            if len(tl) > 0:
                plt.plot(tl, label="train")
            if len(vl) > 0:
                plt.plot(vl, label="val")
            plt.legend()
            plt.title(f"Loss Curves - {name}\nNext-activity")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{name}_loss_curves_reload.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating plot3: {e}")
            plt.close()
        try:
            y_true = ed.get("ground_truth", [])
            y_pred = ed.get("predictions", [])
            if len(y_true) > 0 and len(y_pred) > 0:
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(5, 4))
                plt.imshow(cm, aspect="auto", cmap="Blues")
                plt.colorbar()
                plt.title(f"Confusion Matrix (Test) - {name}\nNext-activity")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(working_dir, f"{name}_confusion_matrix_reload.png")
                )
                plt.close()
        except Exception as e:
            print(f"Error creating plot4: {e}")
            plt.close()
        try:
            pref = ed.get("prefix_lens", [])
            flags = ed.get("top3_flags", [])
            if len(pref) > 0 and len(flags) > 0:
                d = defaultdict(list)
                for L, f in zip(pref, flags):
                    d[int(L)].append(int(f))
                xs = sorted(d.keys())
                ys = [float(np.mean(d[x])) for x in xs]
                plt.figure()
                plt.plot(xs, ys, marker="o")
                plt.title(f"Top-3 Accuracy vs Prefix Length - {name}\nNext-activity")
                plt.xlabel("Prefix Length")
                plt.ylabel("Top-3 Accuracy")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(working_dir, f"{name}_top3_vs_prefixlen_reload.png")
                )
                plt.close()
        except Exception as e:
            print(f"Error creating plot5: {e}")
            plt.close()
        try:
            probs = np.array(ed.get("probs", []))
            y_true = ed.get("ground_truth", [])
            if probs.size > 0 and len(y_true) > 1:
                classes = sorted(set(y_true))
                Y = label_binarize(np.array(y_true), classes=range(probs.shape[1]))
                if len(classes) > 1:
                    grid = np.linspace(0, 1, 101)
                    curves = []
                    aps = []
                    for c in classes:
                        p, r, _ = precision_recall_curve(Y[:, c], probs[:, c])
                        curves.append(np.interp(grid, r[::-1], p[::-1]))
                        aps.append(average_precision_score(Y[:, c], probs[:, c]))
                    macro_p = np.mean(np.stack(curves, 0), 0)
                    plt.figure()
                    plt.plot(grid, macro_p, label=f"mAP={np.mean(aps):.3f}")
                    plt.title(f"Macro Precision-Recall (Test) - {name}\nNext-activity")
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(working_dir, f"{name}_macro_pr_reload.png")
                    )
                    plt.close()
        except Exception as e:
            print(f"Error creating plot6: {e}")
            plt.close()
    # Print evaluation metric summary
    for k, ed in experiment_data_loaded.items():
        tm = dict(ed.get("metrics", {}).get("test", [("final", {})])[0][1])
        if tm:
            print(
                f"{k} | Test acc={tm.get('acc',0):.4f} macro_f1={tm.get('macro_f1',0):.4f} top3={tm.get('top3',0):.4f} loss={tm.get('loss',0):.4f}"
            )


if __name__ == "__main__":
    main()
