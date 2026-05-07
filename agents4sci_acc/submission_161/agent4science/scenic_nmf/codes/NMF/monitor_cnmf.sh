#!/bin/bash
# Monitor cNMF progress

echo "========================================="
echo "cNMF Analysis Progress Monitor"
echo "========================================="
echo ""

# Check if process is running
PID=$(ps aux | grep -E "python.*cnmf_corrected" | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "❌ cNMF process is NOT running"
else
    echo "✅ cNMF process is running (PID: $PID)"
fi

# Count iterations
ITER_COUNT=$(ls cnmf_results_corrected/strict_endocrine_cnmf/cnmf_tmp/ 2>/dev/null | grep iter | wc -l)
TOTAL_ITER=$((41 * 100))
PERCENT=$((ITER_COUNT * 100 / TOTAL_ITER))

echo ""
echo "Progress: $ITER_COUNT / $TOTAL_ITER iterations ($PERCENT%)"
echo ""

# Progress bar
echo -n "["
for i in $(seq 1 50); do
    if [ $i -le $((PERCENT / 2)) ]; then
        echo -n "="
    else
        echo -n " "
    fi
done
echo "] $PERCENT%"

# Estimate remaining time (rough)
if [ $ITER_COUNT -gt 0 ]; then
    UPTIME=$(ps -o etimes= -p $PID 2>/dev/null | xargs)
    if [ ! -z "$UPTIME" ]; then
        RATE=$(echo "scale=2; $ITER_COUNT / $UPTIME" | bc 2>/dev/null)
        REMAINING=$((TOTAL_ITER - ITER_COUNT))
        if [ ! -z "$RATE" ] && [ $(echo "$RATE > 0" | bc) -eq 1 ]; then
            TIME_LEFT=$(echo "scale=0; $REMAINING / $RATE / 60" | bc 2>/dev/null)
            echo ""
            echo "Estimated time remaining: ~${TIME_LEFT} minutes"
        fi
    fi
fi

echo ""
echo "Check details with:"
echo "  tail -f cnmf_corrected.log"
echo ""