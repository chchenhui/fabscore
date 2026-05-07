#!/bin/bash

# Boids Evolution Experiment Runner
# Systematic exploration of emergent intelligence and collaboration
# Author: Automated Research System
# Date: $(date)

echo "🧬 BOIDS EVOLUTION EXPERIMENT SUITE"
echo "======================================"
echo "Running systematic experiments to analyze emergent intelligence..."
echo ""

# Create results directory
mkdir -p experiments/results
mkdir -p experiments/logs

# Experiment configurations
declare -a TOPOLOGIES=("triangle" "line" "star")
declare -a AGENT_COUNTS=(3 4 5 6 8 10)
declare -a STEP_COUNTS=(20 50 100)

# Counter for experiment numbering
EXPERIMENT_ID=1

# Function to run single experiment
run_experiment() {
    local topology=$1
    local agents=$2
    local steps=$3
    local exp_id=$4
    
    echo "🔬 Experiment ${exp_id}: ${topology} topology, ${agents} agents, ${steps} steps"
    
    # Generate unique filename
    local filename="experiments/results/exp_${exp_id}_${topology}_${agents}agents_${steps}steps.json"
    local logfile="experiments/logs/exp_${exp_id}_${topology}_${agents}agents_${steps}steps.log"
    
    # Run simulation with full logging
    python3 main_simple.py \
        --topology ${topology} \
        --agents ${agents} \
        --steps ${steps} \
        --export ${filename} \
        --quiet > ${logfile} 2>&1
    
    if [ $? -eq 0 ]; then
        echo "   ✅ Success: Results saved to ${filename}"
    else
        echo "   ❌ Failed: Check ${logfile} for errors"
    fi
    
    # Brief pause to avoid overwhelming system
    sleep 0.5
}

# Function to run baseline experiments (quick validation)
run_baseline_experiments() {
    echo "📊 PHASE 1: BASELINE EXPERIMENTS"
    echo "Testing basic functionality across configurations..."
    echo ""
    
    # Quick baseline: each topology with moderate settings
    for topology in "${TOPOLOGIES[@]}"; do
        run_experiment ${topology} 4 30 ${EXPERIMENT_ID}
        ((EXPERIMENT_ID++))
    done
    
    echo ""
}

# Function to run scalability experiments
run_scalability_experiments() {
    echo "📈 PHASE 2: SCALABILITY EXPERIMENTS"  
    echo "Testing how patterns change with agent count..."
    echo ""
    
    # Fix topology=triangle, vary agent count
    for agents in "${AGENT_COUNTS[@]}"; do
        run_experiment "triangle" ${agents} 50 ${EXPERIMENT_ID}
        ((EXPERIMENT_ID++))
    done
    
    echo ""
}

# Function to run temporal dynamics experiments
run_temporal_experiments() {
    echo "⏰ PHASE 3: TEMPORAL DYNAMICS EXPERIMENTS"
    echo "Testing how patterns evolve over longer time periods..."
    echo ""
    
    # Fix topology=triangle, agents=5, vary time
    for steps in "${STEP_COUNTS[@]}"; do
        run_experiment "triangle" 5 ${steps} ${EXPERIMENT_ID}
        ((EXPERIMENT_ID++))
    done
    
    echo ""
}

# Function to run network topology experiments
run_topology_experiments() {
    echo "🌐 PHASE 4: NETWORK TOPOLOGY EXPERIMENTS"
    echo "Testing how network structure affects emergence..."
    echo ""
    
    # Fix agents=6, steps=75, vary topology
    for topology in "${TOPOLOGIES[@]}"; do
        run_experiment ${topology} 6 75 ${EXPERIMENT_ID}
        ((EXPERIMENT_ID++))
        
        # Also test with more agents for this topology
        run_experiment ${topology} 8 75 ${EXPERIMENT_ID}
        ((EXPERIMENT_ID++))
    done
    
    echo ""
}

# Function to run intensive collaboration experiments
run_collaboration_experiments() {
    echo "🤝 PHASE 5: COLLABORATION INTENSIVE EXPERIMENTS"
    echo "Testing conditions that maximize collaborative patterns..."
    echo ""
    
    # Longer simulations with moderate agent counts to see deep patterns
    for topology in "${TOPOLOGIES[@]}"; do
        run_experiment ${topology} 5 150 ${EXPERIMENT_ID}
        ((EXPERIMENT_ID++))
        
        run_experiment ${topology} 7 100 ${EXPERIMENT_ID}
        ((EXPERIMENT_ID++))
    done
    
    echo ""
}

# Function to generate summary report
generate_summary() {
    echo "📋 EXPERIMENT SUMMARY"
    echo "===================="
    echo "Total experiments run: $((EXPERIMENT_ID - 1))"
    echo "Results directory: experiments/results/"
    echo "Logs directory: experiments/logs/"
    echo ""
    echo "Next steps:"
    echo "1. Run: python3 analyze_experiments.py"
    echo "2. Check: experimentation_analysis.md"
    echo ""
}

# Main execution
main() {
    echo "Starting experiment suite at $(date)"
    echo ""
    
    # Check if main_simple.py exists
    if [ ! -f "main_simple.py" ]; then
        echo "❌ Error: main_simple.py not found in current directory"
        echo "Please run this script from the boids-evolution project root"
        exit 1
    fi
    
    # Run all experiment phases
    run_baseline_experiments
    run_scalability_experiments  
    run_temporal_experiments
    run_topology_experiments
    run_collaboration_experiments
    
    # Generate summary
    generate_summary
    
    echo "🎉 All experiments completed at $(date)"
    echo ""
    echo "🔍 Quick Analysis Preview:"
    echo "=========================="
    
    # Count total tools created across all experiments
    echo "Analyzing results..."
    if command -v jq &> /dev/null; then
        total_experiments=$(ls experiments/results/*.json 2>/dev/null | wc -l)
        if [ ${total_experiments} -gt 0 ]; then
            total_tools=$(jq -s 'map(.final_state.total_tools) | add' experiments/results/*.json 2>/dev/null || echo "N/A")
            avg_specializations=$(jq -s 'map(.final_state.agent_specializations | length) | add / length' experiments/results/*.json 2>/dev/null || echo "N/A")
            
            echo "   📊 Total experiments: ${total_experiments}"
            echo "   🔧 Total tools created: ${total_tools}"
            echo "   🎯 Avg specialization types: ${avg_specializations}"
        fi
    else
        echo "   📊 Install 'jq' for detailed JSON analysis"
    fi
    
    echo ""
    echo "Run 'python3 analyze_experiments.py' for detailed analysis!"
}

# Handle command line arguments
case "${1:-all}" in
    "baseline")
        run_baseline_experiments
        ;;
    "scalability") 
        run_scalability_experiments
        ;;
    "temporal")
        run_temporal_experiments
        ;;
    "topology")
        run_topology_experiments
        ;;
    "collaboration")
        run_collaboration_experiments
        ;;
    "all")
        main
        ;;
    *)
        echo "Usage: $0 [baseline|scalability|temporal|topology|collaboration|all]"
        echo "Default: all"
        exit 1
        ;;
esac 