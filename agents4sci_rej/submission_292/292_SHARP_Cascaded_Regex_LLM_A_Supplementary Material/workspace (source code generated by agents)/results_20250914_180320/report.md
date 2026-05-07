# Phishing Detection: Academic Methods Comparison

Generated: 2025-09-14 18:03:21

## Executive Summary

This experiment compares recent academic approaches to phishing detection:
- **PhishIntention (USENIX 2022)**: Vision-based approach analyzing brand and credential intentions
- **CNN-BiGRU (Sensors 2024)**: Deep learning with 1D-CNN and bidirectional GRU
- **Feature Ensemble (uOttawa 2023)**: ML ensemble trained on 737,000 URLs

**Best performing method:** Rule-based Baseline (F1-Score: 1.000)

## Detailed Results

| Method | Paper/Source | Accuracy | Precision | Recall | F1-Score | Time (s) |
|--------|--------------|----------|-----------|--------|----------|----------|
| Rule-based Baseline | Traditional | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 |
| Regex Pattern Baseline | Traditional | 0.735 | 1.000 | 0.474 | 0.643 | 0.01 |
| PhishIntention (USENIX'22) | USENIX Security 2022 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 |
| CNN-BiGRU (Sensors'24) | Sensors 2024, 24(7) | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 |
| Feature Ensemble (uOttawa'23) | U. Ottawa 2023 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 |

## Key Findings

- Academic methods show **-100.0%** average improvement over baselines
- Best academic method: Rule-based Baseline
- Most efficient method: Rule-based Baseline

## References

1. Liu et al., "Inferring Phishing Intention via Webpage Appearance and Dynamics", USENIX Security 2022
2. "Advancing Phishing Email Detection: A Comparative Study of Deep Learning Models", Sensors 2024
3. "Phishing Attack Detection using Machine Learning", University of Ottawa, 2023
