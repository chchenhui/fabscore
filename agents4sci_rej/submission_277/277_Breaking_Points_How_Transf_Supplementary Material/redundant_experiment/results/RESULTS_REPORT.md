# Compression Circuit Analysis Results

**Date**: 2025-09-14 23:41
**Model**: GPT-2
**Total Circuits Found**: 31

## Summary Statistics

- **Attention Circuits**: 31
- **MLP Circuits**: 0
- **Average Importance Score**: 0.6874
- **Max Importance Score**: 1.2615

## Top 5 Compression Circuits

| Rank | Circuit | Type | Importance Score |
|------|---------|------|------------------|
| 1 | L4H8 | Attention | 1.2615 |
| 2 | L5H0 | Attention | 1.0245 |
| 3 | L6H10 | Attention | 1.0047 |
| 4 | L8H11 | Attention | 0.9610 |
| 5 | L9H11 | Attention | 0.9323 |


## Layer Distribution

- Layer 0: 6 circuits
- Layer 1: 4 circuits
- Layer 2: 1 circuits
- Layer 4: 2 circuits
- Layer 5: 1 circuits
- Layer 6: 2 circuits
- Layer 7: 4 circuits
- Layer 8: 4 circuits
- Layer 9: 2 circuits
- Layer 10: 1 circuits
- Layer 11: 4 circuits


## Figures Generated

1. **compression_circuits_main.png**: Main visualization with heatmaps and distributions
2. **analysis_summary.png**: Comprehensive summary statistics
3. **attention_head_heatmap.png**: Detailed attention head importance map
4. **circuit_specialization.png**: Circuit specialization scores

## Key Findings

1. Compression circuits are distributed across multiple layers
2. Both attention and MLP components contribute to compression
3. Circuits show specialization for different types of redundancy

## Next Steps

- Ablation studies to verify causal role
- Test on larger models (GPT-2-medium, GPT-2-large)
- Apply to real-world redundant data (code, structured documents)
