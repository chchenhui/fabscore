# In-Domain Drift Adapter Training and Evaluation

## Experiment Overview

Trained and evaluated in-domain drift adapters as the upper bound for adapter-based embedding model upgrades. For each of 4 BEIR datasets, a ResidualMLPAdapter was trained on N_p=5000 paired embeddings sampled from the target corpus documents, mapping f_new (all-mpnet-base-v2) embeddings to f_old (all-distilroberta-v1) space using MSE loss. Each adapter was trained with 3 random seeds [0, 1, 2].

## Setup

- **Models**: f_old = all-distilroberta-v1, f_new = all-mpnet-base-v2 (both 768-dim)
- **Datasets**: SciFact (5183 corpus), TREC-COVID (171332), FiQA-2018 (57638), ArguAna (8674)
- **Adapter**: ResidualMLPAdapter(embed_dim=768, hidden_dim=256)
- **Training**: N_p=5000 paired embeddings (or all docs if < 5000), 80/20 train/val split, AdamW lr=3e-4, weight_decay=0.01, batch_size=256, max 50 epochs, early stopping patience=5
- **Evaluation**: Adapted queries retrieved against corpus_fold using FAISS FlatIP top-100, nDCG@10 and Recall@10 via BEIR

## Key Results

| Dataset    | nDCG@10 (mean +/- std) | Recall@10 (mean +/- std) | Oracle nDCG@10 | Recovery % |
|------------|------------------------|--------------------------|----------------|------------|
| SciFact    | 0.4438 +/- 0.0059     | 0.5973 +/- 0.0106        | 0.6557         | 67.7%      |
| TREC-COVID | 0.3365 +/- 0.0064     | 0.0083 +/- 0.0002        | 0.5133         | 65.6%      |
| FiQA-2018  | 0.2303 +/- 0.0036     | 0.3050 +/- 0.0081        | 0.4996         | 46.1%      |
| ArguAna    | 0.3900 +/- 0.0009     | 0.6771 +/- 0.0041        | 0.4652         | 83.8%      |

Recovery % = in-domain nDCG@10 / oracle nDCG@10 (misaligned is ~0 for all datasets).

## Key Observations

1. All training runs hit the max 50 epochs without early stopping, suggesting the loss was still decreasing. The adapter architecture is small enough that overfitting is not a concern at this scale.
2. Low variance across seeds (std < 0.01 nDCG@10 for all datasets), indicating stable training.
3. The in-domain adapter recovers 46-84% of oracle performance depending on the dataset, establishing a meaningful upper bound for the adapter approach.
4. SciFact used all 5000 of its 5183 corpus docs for sampling (near full-corpus coverage), while other datasets sampled from much larger corpora.
5. Training was very fast (~3-5 seconds per run on GPU), confirming the lightweight nature of the adapter.

## Files

- Results: `pada/results/in_domain.json`
- Checkpoints: `pada/outputs/in_domain/{dataset}/seed_{s}/best_adapter.pt`
- Training histories: `pada/outputs/in_domain/{dataset}/seed_{s}/training_history.json`
- Sampled doc IDs: `pada/outputs/in_domain/{dataset}/seed_{s}/sampled_doc_ids.json`
- WandB logs: `wandb/offline-run-20260228_110731-cujadq6h`
