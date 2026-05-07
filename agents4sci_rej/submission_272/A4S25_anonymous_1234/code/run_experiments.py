
import argparse
import hashlib, json, os, random, pathlib
import numpy as np
import pandas as pd


def _stable_unit_float(s: str, mod: int = 1000) -> float:
    """Stable across runs & platforms (md5 modulo)."""
    h = hashlib.md5(str(s).encode("utf-8")).hexdigest()
    return (int(h, 16) % mod) / float(mod)
from sklearn.metrics import roc_auc_score, average_precision_score

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def generate_synthetic_graph(n_diseases=150, n_genes=200, n_drugs=80, n_symptoms=120, p_edge=0.05):
    # Create typed node IDs
    diseases = [f"D{i}" for i in range(n_diseases)]
    genes = [f"G{i}" for i in range(n_genes)]
    drugs = [f"R{i}" for i in range(n_drugs)]
    symptoms = [f"S{i}" for i in range(n_symptoms)]
    # Relations
    rels = []
    # disease-treated_by-drug
    for d in diseases:
        for r in drugs:
            if np.random.rand() < p_edge:
                rels.append((d, "treated_by", r))
    # disease-associated_with-gene
    for d in diseases:
        for g in genes:
            if np.random.rand() < p_edge:
                rels.append((d, "associated_with", g))
    # disease-has_symptom-symptom
    for d in diseases:
        for s in symptoms:
            if np.random.rand() < p_edge:
                rels.append((d, "has_symptom", s))
    return rels, diseases, genes, drugs, symptoms

def train_dummy_link_predictor(edges, all_heads, all_tails, neg_ratio=1.0):
    # Build positives and negatives for a simple binary classification on (h, r, t)
    pos = pd.DataFrame(edges, columns=["h","r","t"])
    # Sample negatives from non-existing pairs for each relation type
    rel_types = pos["r"].unique().tolist()
    neg_rows = []
    for r in rel_types:
        sub = pos[pos.r == r]
        heads = sub["h"].unique().tolist()
        tails = sub["t"].unique().tolist()
        # Build candidate non-edges
        candidates = []
        for h in heads:
            for t in tails:
                candidates.append((h, r, t))
        pos_set = set(map(tuple, sub[["h","r","t"]].values.tolist()))
        non_edges = [c for c in candidates if c not in pos_set]
        k = min(len(sub), int(len(non_edges) * 0.1) + 1)  # sample some negatives
        chosen = random.sample(non_edges, k) if non_edges else []
        neg_rows.extend(chosen)
    neg = pd.DataFrame(neg_rows, columns=["h","r","t"])
    pos["y"] = 1
    neg["y"] = 0
    data = pd.concat([pos, neg], ignore_index=True)
    # Simple scoring: hash-based deterministic pseudo-embedding similarity
    def score_row(row):
        # Deterministic pseudo-random based on strings
        h_val = (_stable_unit_float(row['h']) % 1000) / 1000.0
        t_val = (_stable_unit_float(row['t']) % 1000) / 1000.0
        r_bias = (_stable_unit_float(row['r'], mod=100) % 100) / 100.0
        return 0.5*h_val + 0.5*t_val + 0.1*r_bias
    data["score"] = data.apply(score_row, axis=1)
    return data

def compute_metrics(df):
    y_true = df["y"].values
    scores = df["score"].values
    try:
        auroc = roc_auc_score(y_true, scores)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, scores)
    except Exception:
        auprc = float("nan")
    # Hit@10: fraction of positives ranked in top 10% within each relation bucket
    hit_counts = 0
    total_pos = 0
    for r, group in df.groupby("r"):
        group = group.sort_values("score", ascending=False)
        top_k = max(1, int(0.10 * len(group)))
        top = group.head(top_k)
        hit_counts += top["y"].sum()
        total_pos += group["y"].sum()
    hit_at_10 = (hit_counts / total_pos) if total_pos > 0 else 0.0
    return {"AUROC": float(auroc), "AUPRC": float(auprc), "Hit@10": float(hit_at_10)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--results-dir", type=str, default="results")
    args = ap.parse_args()

    set_seed(args.random_seed)

    edges, diseases, genes, drugs, symptoms = generate_synthetic_graph()
    df = train_dummy_link_predictor(edges, diseases+genes+drugs+symptoms, diseases+genes+drugs+symptoms)
    metrics = compute_metrics(df)

    # Ensure directories
    results_dir = pathlib.Path(args.results_dir)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)

    # Write predictions
    df.to_csv(results_dir / "predictions.csv", index=False)

    # Write metrics.json
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Simple tables
    # Per-relation metrics (AUROC only for brevity)
    rel_rows = []
    for r, group in df.groupby("r"):
        try:
            auroc_r = roc_auc_score(group["y"].values, group["score"].values)
        except Exception:
            auroc_r = float("nan")
        rel_rows.append({"relation": r, "AUROC": float(auroc_r), "n": int(len(group))})
    pd.DataFrame(rel_rows).to_csv(results_dir / "tables" / "per_relation.csv", index=False)

    print("Finished. Metrics:", metrics)

if __name__ == "__main__":
    main()
