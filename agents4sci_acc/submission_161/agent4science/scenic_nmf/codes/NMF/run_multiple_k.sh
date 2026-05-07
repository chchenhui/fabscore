#!/bin/bash
# Script to run NMF analysis with different k values

# Activate the conda environment
source /wanglab/jguo/miniconda3/bin/activate NMF

# Test different k values
for k in 10 15 20 25 30
do
    echo "========================================="
    echo "Running NMF with k=$k"
    echo "========================================="
    python run_nmf_complete.py --k $k --output nmf_results
    echo "Completed k=$k"
    echo ""
done

echo "All analyses complete!"
echo "Results are in:"
for k in 10 15 20 25 30
do
    echo "  - nmf_results_k$k/"
done