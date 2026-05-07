# Synthetic Rare Disease Knowledge Graph (Data Card)

**Dataset name:** synthetic_rare_disease_KG  
**Version:** 1.0 (seeded)  
**License:** CC0 (public domain)  
**Provenance:** Generated locally by `code/run_experiments.py` when first run.

## Generation Procedure
- Nodes: diseases, genes, drugs, and symptoms.
- Relations: (disease, treated_by, drug), (disease, associated_with, gene), (disease, has_symptom, symptom).
- Graph generated deterministically given `--random-seed` (default 42).

## Splits
- Train/Validation/Test edges split by stratified random partition with fixed seed.
- Negative edges sampled uniformly from non-edges.

## PII / Safety
- Entirely synthetic. No PII. No regulated data.

## Intended Use
- Reproducible link prediction baselines.
- Sanity-check metrics (AUROC, AUPRC, Hit@10) on synthetic data.

## Limitations
- Synthetic structure may not reflect real-world frequency or noise.
- Metrics are not indicative of clinical utility.

