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
from ai_scientist.ideas.my_research_topic import load_datasets

# Device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experiment data
experiment_data = {
    "BPI2012": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "workload": {"val": [], "test": []},
    },
    "BPI2017": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "workload": {"val": [], "test": []},
    },
    "ROAD": {
        "metrics": {"train": [], "val": [], "test": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "workload": {"val": [], "test": []},
    },
}


# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# Time-based split
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


# Prefix builder with process-aware features
def build_prefix_dataset(df, max_prefix_len=10, min_prefix_len=1):
    df = df.copy()
    if "lifecycle" in df.columns:
        df = df[df["lifecycle"].astype(str).str.lower().eq("complete")]
        if df.empty:
            df = df.copy()
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
        ts = pd.to_datetime(g["timestamp"], utc=True)
        ts_sec = (ts.astype("int64") // 10**9).to_numpy(np.int64)
        acts_ids = np.array(
            [act2id[a] for a in g["activity"].astype(str)], dtype=np.int64
        )
        hours = (ts.dt.hour.to_numpy(dtype=float) / 23.0).astype(np.float32)
        weekdays = (ts.dt.weekday.to_numpy(dtype=float) / 6.0).astype(np.float32)
        working = (
            (
                (ts.dt.weekday.to_numpy() < 5)
                & (ts.dt.hour.to_numpy() >= 8)
                & (ts.dt.hour.to_numpy() <= 17)
            )
        ).astype(np.float32)
        deltas = np.diff(ts_sec, prepend=ts_sec[0]).astype(np.float32)
        since_start = (ts_sec - ts_sec[0]).astype(np.float32)
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
                    "last_ts": int(ts_sec[k - 1]),
                    "next_ts": int(ts_sec[k]),
                }
            )
    # normalize time features globally first (refined later on train)
    if len(samples) > 0:
        all_feats = np.concatenate(
            [s["seq_feats"] for s in samples if len(s["seq_feats"]) > 0], axis=0
        )
        for col in [0, 1]:
            mu, sigma = all_feats[:, col].mean(), all_feats[:, col].std() + 1e-6
            for s in samples:
                s["seq_feats"][:, col] = (s["seq_feats"][:, col] - mu) / sigma
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
            [np.zeros((pad, self.num_cont), dtype=np.float32), feats.astype(np.float32)]
        )
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

    def forward(self, acts, feats, mask):
        x = self.emb(acts)
        x = torch.cat([x, feats], dim=-1)
        out, (h, c) = self.lstm(x)
        h_last = self.dropout(h[-1])
        return self.fc(h_last)


def evaluate(model, loader, criterion, device, num_classes, pad_idx):
    model.eval()
    total_loss = 0.0
    ys = []
    preds = []
    probs_list = []
    top3_correct = 0
    N = 0
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
            _, topk = torch.topk(probs, k=k_val, dim=1)
            for i in range(batch["y"].size(0)):
                if batch["y"][i].item() in topk[i].detach().cpu().tolist():
                    top3_correct += 1
            ys.extend(batch["y"].detach().cpu().tolist())
            preds.extend(top1.detach().cpu().tolist())
            probs_list.append(probs.detach().cpu().numpy())
            N += logits.size(0)
    avg_loss = total_loss / max(1, N)
    y_true = np.array(ys)
    y_pred = np.array(preds)
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    try:
        f1 = float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        f1 = 0.0
    top3 = float(top3_correct / max(1, N))
    probs_concat = (
        np.concatenate(probs_list, axis=0)
        if len(probs_list) > 0
        else np.zeros((0, num_classes + 1))
    )
    return avg_loss, acc, f1, top3, y_true, y_pred, probs_concat


# Resource workload model: Poisson rates per resource per hour bin
def estimate_resource_rates(train_df, hour_bins=24):
    df = train_df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = df["ts"].dt.hour
    resources = df["resource"].astype(str).fillna("System").tolist()
    df["resource"] = resources
    # time span in days
    tmin = df["ts"].min()
    tmax = df["ts"].max()
    span_days = max(1, int(np.ceil((tmax - tmin).total_seconds() / 86400.0)))
    rate = {}  # rate[res][hour] = events/sec
    for res, g in df.groupby("resource"):
        rate[res] = {}
        counts = g.groupby("hour").size()
        for h in range(hour_bins):
            c = int(counts.get(h, 0))
            # expected duration per bin over span_days: 3600 * span_days seconds
            lam = c / (3600.0 * span_days)
            rate[res][h] = lam
    # global fallback
    global_counts = df.groupby("hour").size()
    global_rate = {
        h: int(global_counts.get(h, 0)) / (3600.0 * span_days) for h in range(hour_bins)
    }
    return rate, global_rate


def forecast_workload_mc(
    prefix_samples, split_df, rates, global_rate, H=3600, M=20, hour_bins=24
):
    # Prepare realized counts per prefix
    split_df = split_df.copy()
    split_df["ts_sec"] = (
        pd.to_datetime(split_df["timestamp"], utc=True).astype("int64") // 10**9
    ).astype(np.int64)
    resources = split_df["resource"].astype(str).fillna("System")
    split_df["resource"] = resources
    res_list = sorted(split_df["resource"].unique().tolist())
    forecasts = []
    actuals = []
    for s in prefix_samples:
        t0 = s["last_ts"]
        t1 = t0 + H
        h = int((pd.to_datetime(t0, unit="s", utc=True).hour) % hour_bins)
        # Monte Carlo simulate counts per resource assuming Poisson arrivals
        mc_sum = {r: 0.0 for r in res_list}
        for _ in range(M):
            for r in res_list:
                lam = rates.get(r, {}).get(h, global_rate.get(h, 0.0))
                cnt = np.random.poisson(lam * H)
                mc_sum[r] += cnt
        forecast = {r: mc_sum[r] / M for r in res_list}
        # Actual realized counts in (t0, t1]
        window = split_df[(split_df["ts_sec"] > t0) & (split_df["ts_sec"] <= t1)]
        act_counts = window.groupby("resource").size().to_dict()
        actual = {r: int(act_counts.get(r, 0)) for r in res_list}
        forecasts.append(forecast)
        actuals.append(actual)
    # Compute wMAPE across resources aggregated over prefixes
    F = np.array([[f[r] for r in res_list] for f in forecasts], dtype=float)
    A = np.array([[a[r] for r in res_list] for a in actuals], dtype=float)
    abs_err = np.abs(F - A)
    total_actual = np.sum(A)
    wmape = (
        float(np.sum(abs_err) / (total_actual + 1e-8))
        if total_actual > 0
        else float(np.sum(abs_err) / (np.sum(F) + 1e-8))
    )
    return wmape, res_list, F, A


def train_one_dataset(
    name, df, max_epochs=8, batch_size=128, max_prefix_len=10, lr=1e-3
):
    print(f"\n=== Dataset: {name} ===")
    # time split
    train_cases, val_cases, test_cases = time_based_split(df, 0.7, 0.15)
    # build samples and vocab
    samples_all, act2id, id2act, pad_id = build_prefix_dataset(
        df, max_prefix_len=max_prefix_len
    )
    samples_train = [s for s in samples_all if s["case_id"] in train_cases]
    samples_val = [s for s in samples_all if s["case_id"] in val_cases]
    samples_test = [s for s in samples_all if s["case_id"] in test_cases]
    # refine normalization on train
    if len(samples_train) > 0:
        feats = [s["seq_feats"] for s in samples_train if s["seq_feats"].shape[0] > 0]
        if len(feats) > 0:
            allf = np.concatenate(feats, axis=0)
            for col in [0, 1]:
                mu, sigma = allf[:, col].mean(), allf[:, col].std() + 1e-6
                for arr in (samples_train, samples_val, samples_test):
                    for s in arr:
                        s["seq_feats"][:, col] = (s["seq_feats"][:, col] - mu) / sigma
    print(
        f"Samples train/val/test: {len(samples_train)}/{len(samples_val)}/{len(samples_test)}; vocab={len(act2id)}"
    )
    if len(samples_train) == 0 or len(act2id) < 2:
        print("Not enough data, skipping.")
        return

    ds_train = PrefixDataset(samples_train, pad_id, max_len=max_prefix_len, num_cont=5)
    ds_val = PrefixDataset(samples_val, pad_id, max_len=max_prefix_len, num_cont=5)
    ds_test = PrefixDataset(samples_test, pad_id, max_len=max_prefix_len, num_cont=5)
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

    # model
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

    # Resource rates from train portion of raw df
    df_train = df[df["case_id"].isin(train_cases)].copy()
    df_val = df[df["case_id"].isin(val_cases)].copy()
    df_test = df[df["case_id"].isin(test_cases)].copy()
    rates, global_rate = estimate_resource_rates(df_train, hour_bins=24)
    H = 3600
    M = 20

    # training loop
    best_val_top3 = -1.0
    best_state = None
    val_wmape_per_epoch = []
    val_loss_per_epoch = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0
        run_loss = 0.0
        for batch in dl_train:
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
        train_loss = run_loss / max(1, total)
        # evaluate val
        val_loss, val_acc, val_f1, val_top3, _, _, _ = evaluate(
            model, dl_val, criterion, device, len(act2id), pad_id
        )
        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
        # resource workload wMAPE@H on val (Monte Carlo Poisson)
        wmape_val, res_list_val, Fv, Av = forecast_workload_mc(
            samples_val, df_val, rates, global_rate, H=H, M=M
        )
        print(
            f"  Val: acc={val_acc:.4f} macroF1={val_f1:.4f} top3={val_top3:.4f} | wMAPE@H={wmape_val:.4f}"
        )
        # save per-epoch
        experiment_data[name]["losses"]["train"].append((epoch, train_loss))
        experiment_data[name]["losses"]["val"].append((epoch, val_loss))
        experiment_data[name]["metrics"]["val"].append(
            (
                epoch,
                {
                    "acc": val_acc,
                    "macro_f1": val_f1,
                    "top3": val_top3,
                    "wMAPE@H": wmape_val,
                },
            )
        )
        experiment_data[name]["epochs"].append(epoch)
        val_wmape_per_epoch.append(wmape_val)
        val_loss_per_epoch.append(val_loss)
        if val_top3 > best_val_top3:
            best_val_top3 = val_top3
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # final eval
    train_loss, train_acc, train_f1, train_top3, _, _, _ = evaluate(
        model, dl_train, criterion, device, len(act2id), pad_id
    )
    val_loss, val_acc, val_f1, val_top3, _, _, _ = evaluate(
        model, dl_val, criterion, device, len(act2id), pad_id
    )
    test_loss, test_acc, test_f1, test_top3, y_true_t, y_pred_t, probs_t = evaluate(
        model, dl_test, criterion, device, len(act2id), pad_id
    )
    wmape_val, res_list_val, Fv, Av = forecast_workload_mc(
        samples_val, df_val, rates, global_rate, H=H, M=M
    )
    wmape_test, res_list_test, Ft, At = forecast_workload_mc(
        samples_test, df_test, rates, global_rate, H=H, M=M
    )

    print(
        f"[{name}] Train: loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} top3={train_top3:.4f}"
    )
    print(
        f"[{name}] Val:   loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} top3={val_top3:.4f} wMAPE@H={wmape_val:.4f}"
    )
    print(
        f"[{name}] Test:  loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f} top3={test_top3:.4f} wMAPE@H={wmape_test:.4f}"
    )

    experiment_data[name]["metrics"]["train"].append(
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
    experiment_data[name]["metrics"]["val"].append(
        (
            "final",
            {
                "loss": val_loss,
                "acc": val_acc,
                "macro_f1": val_f1,
                "top3": val_top3,
                "wMAPE@H": wmape_val,
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
                "wMAPE@H": wmape_test,
            },
        )
    )
    experiment_data[name]["predictions"] = y_pred_t.tolist()
    experiment_data[name]["ground_truth"] = y_true_t.tolist()
    experiment_data[name]["workload"]["val"] = {
        "resources": res_list_val,
        "forecast": Fv,
        "actual": Av,
    }
    experiment_data[name]["workload"]["test"] = {
        "resources": res_list_test,
        "forecast": Ft,
        "actual": At,
    }

    # plots
    try:
        plt.figure()
        plt.plot(
            [t for (_, t) in experiment_data[name]["losses"]["train"]],
            label="train_loss",
        )
        plt.plot(
            [v for (_, v) in experiment_data[name]["losses"]["val"]], label="val_loss"
        )
        plt.legend()
        plt.title(f"Loss Curves - {name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{name}.png"))
        plt.close()

        plt.figure()
        plt.plot(val_wmape_per_epoch, label="val_wMAPE@H")
        plt.legend()
        plt.title(f"Val wMAPE@H - {name}")
        plt.xlabel("Epoch")
        plt.ylabel("wMAPE")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"wmape_val_{name}.png"))
        plt.close()

        # scatter of workload forecast vs actual (test)
        if Ft.size > 0:
            plt.figure()
            plt.scatter(At.flatten(), Ft.flatten(), alpha=0.3)
            plt.xlabel("Actual count (test, next H)")
            plt.ylabel("Forecast count")
            plt.title(f"Resource workload scatter - {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"workload_scatter_{name}.png"))
            plt.close()
    except Exception as e:
        print(f"[warn] Plotting failed: {e}")

    # save matrices
    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true_t, y_pred_t)
        np.save(os.path.join(working_dir, f"cm_{name}.npy"), cm)
        np.save(
            os.path.join(working_dir, f"workload_val_{name}.npy"),
            {"resources": res_list_val, "forecast": Fv, "actual": Av},
            allow_pickle=True,
        )
        np.save(
            os.path.join(working_dir, f"workload_test_{name}.npy"),
            {"resources": res_list_test, "forecast": Ft, "actual": At},
            allow_pickle=True,
        )
    except Exception as e:
        print(f"[warn] Saving arrays failed: {e}")


def main():
    datasets = load_datasets()
    # limit cases if too large
    for key, df in datasets.items():
        try:
            starts = (
                df.sort_values("timestamp")
                .groupby("case_id")["timestamp"]
                .min()
                .reset_index()
            )
            if len(starts) > 5000:
                keep = set(starts.iloc[:5000]["case_id"])
                df_use = df[df["case_id"].isin(keep)].copy()
            else:
                df_use = df
        except Exception:
            df_use = df
        train_one_dataset(
            key, df_use, max_epochs=8, batch_size=128, max_prefix_len=10, lr=1e-3
        )
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    np.savez_compressed(
        os.path.join(working_dir, "experiment_data_compressed.npz"),
        data=experiment_data,
    )


# Execute immediately
main()
