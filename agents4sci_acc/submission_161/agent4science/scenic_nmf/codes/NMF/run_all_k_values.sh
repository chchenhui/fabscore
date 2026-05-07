#!/bin/bash
# Script to run NMF analysis for k=10 to k=50

# Activate the conda environment
source /wanglab/jguo/miniconda3/bin/activate NMF

# Start time
start_time=$(date +%s)
echo "========================================="
echo "Starting NMF analysis for k=10 to k=50"
echo "Start time: $(date)"
echo "========================================="

# Run for all k values from 10 to 50
for k in {10..50}
do
    echo ""
    echo "========================================="
    echo "Running NMF with k=$k"
    echo "Time: $(date)"
    echo "========================================="
    
    # Run the analysis
    python run_nmf_complete.py --k $k --output nmf_results
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed k=$k"
    else
        echo "✗ Error occurred for k=$k"
    fi
done

# End time
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "========================================="
echo "ALL ANALYSES COMPLETE!"
echo "End time: $(date)"
echo "Total duration: $((duration / 60)) minutes"
echo "========================================="
echo ""
echo "Results are saved in:"
for k in {10..50}
do
    if [ -d "nmf_results_k$k" ]; then
        echo "  ✓ nmf_results_k$k/"
    else
        echo "  ✗ nmf_results_k$k/ (missing)"
    fi
done

echo ""
echo "To compare results across k values, run:"
echo "python compare_k_values.py --k-values $(seq 10 50 | tr '\n' ' ')"