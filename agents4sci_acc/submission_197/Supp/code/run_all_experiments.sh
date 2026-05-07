#!/bin/bash

# ==============================================================================
# Boids-Evolution Experiment Runner - Parallel Execution Script
# ==============================================================================
# This script runs the standard experiment configuration for all 10 meta-prompt
# scenarios in parallel to maximize efficiency.
#
# Standard Configuration: 20 Agents, 15 Rounds
# ==============================================================================

# Activate the project's virtual environment to ensure correct dependencies
source venv/bin/activate

echo "🚀 Launching all 10 experiments in parallel..."
echo "Standard Configuration: 20 Agents, 15 Rounds"
echo "Monitor individual experiment logs for progress. This will take a while."
echo "----------------------------------------------------------------------"

# Define the standard experiment parameters
NUM_AGENTS=20
NUM_ROUNDS=15

# Array of all 10 meta-prompt IDs from meta_prompts.json
SCENARIOS=(
    "data_science_suite"
    "creative_writing_assistant"
    "code_generation_toolkit"
    "file_system_organizer"
    "text_analysis_tools"
    "research_assistant_bot"
    "image_processing_kit"
    "personal_finance_manager"
    "web_scraping_utilities"
    "simulation_and_modeling"
)

# Loop through each scenario and launch the experiment as a background process
for SCENARIO_ID in "${SCENARIOS[@]}"
do
    echo "   ▶️  Starting experiment for scenario: $SCENARIO_ID"
    python3 run_real_experiment.py \
        --meta_prompt_id "$SCENARIO_ID" \
        --num_agents $NUM_AGENTS \
        --num_rounds $NUM_ROUNDS &
done

# The 'wait' command is crucial. It tells the script to wait until all
# background jobs (the experiments launched with '&') are finished.
echo "----------------------------------------------------------------------"
echo "✅ All experiments launched. Waiting for all processes to complete..."

wait

echo "🎉 All 10 experiments have completed successfully!"
echo "----------------------------------------------------------------------" 