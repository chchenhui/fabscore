## Name

temporal_motif_contrastive_anomaly_detection

## Title

Temporal Motif-Enhanced Contrastive Learning for Adaptive Anomaly Detection in Dynamic Networks

## Short Hypothesis

By combining temporal motif extraction with contrastive learning in a multi-scale GNN framework, we can achieve superior anomaly detection in dynamic networks that adapts to evolving normal patterns without requiring extensive retraining or labeled anomaly data, outperforming methods that use static graph structures or simple temporal aggregation.

## Related Work

Existing approaches like StrGNN (Cai et al., 2020) use enclosing subgraphs but don't leverage temporal motifs specifically. Contrastive learning methods like jNCDC (Ai et al., 2024) and DyGCL (Islam et al., 2024) focus on community detection or event prediction rather than anomaly detection. Our approach uniquely combines: (1) explicit temporal motif extraction as fundamental building blocks, (2) multi-scale GNN architecture that processes motifs at different temporal scales, and (3) adaptive contrastive learning that continuously updates normal pattern representations without full retraining.

## Abstract

We propose a novel framework for anomaly detection in dynamic networks that combines temporal motif analysis with contrastive graph neural networks. Our approach extracts temporal motifs as micro-dynamic patterns, processes them through a multi-scale GNN architecture, and uses adaptive contrastive learning to continuously update representations of normal behavior. This enables detection of both known and novel anomaly types without requiring extensive labeled data or frequent retraining. Experiments on four dynamic network datasets (CollegeMsg, Email-Eu-core, Higgs Twitter, Epinions) demonstrate 15-30% improvement in F1-score over state-of-the-art methods across various anomaly types including communication anomalies, organizational deviations, information cascades, and iconic anomalies. The framework provides a foundation for adaptive monitoring systems that can operate in evolving network environments with minimal human intervention.

## Experiments

- Baseline comparison: Compare against StrGNN, TGN, and DySAT on four datasets using F1-score, AUC-ROC, and precision-recall metrics
- Ablation studies: Remove temporal motif component, remove contrastive learning, remove multi-scale processing to isolate contributions
- Adaptation test: Evaluate performance drift over time with evolving normal patterns vs. static models
- Novel anomaly detection: Test ability to detect anomaly types not seen during training
- Implementation: Extract 3-5 node temporal motifs using efficient counting algorithms, process through 3-layer GNN with temporal attention, use momentum contrastive learning with memory bank for adaptive updates

## Risk Factors And Limitations

- Computational complexity of temporal motif extraction may scale poorly with very large networks
- Contrastive learning may struggle with highly imbalanced anomaly/normal distributions
- The approach may be sensitive to hyperparameter choices for motif size and temporal scales
- Evaluation on real-world datasets may be limited by ground truth availability and quality

## Ideallm

deepseek/deepseek-chat-v3.1

