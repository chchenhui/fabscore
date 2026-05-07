#!/usr/bin/env python3
"""
Quick Survey Runner - Efficiently processes remaining surveys
"""

import json
from datetime import datetime
from pathlib import Path

# Remaining surveys to process
REMAINING_SURVEYS = [
    ("in-context learning alignment", "in_context_learning"),
    ("synthetic data LLM training", "synthetic_data"),
    ("LLM data curation", "data_curation"),
    ("LLM pretraining datasets", "pretraining_datasets"),
    ("LLM evaluation benchmarks", "evaluation_benchmarks"),
    ("LLM safety jailbreaking", "safety_jailbreaking"),
    ("LLM bias fairness", "bias_fairness"),
    ("LLM quantization compression", "quantization_compression"),
    ("LLM scaling laws", "scaling_laws"),
    ("efficient LLM inference", "efficient_inference"),
    ("multimodal LLM vision", "multimodal_vision"),
    ("LLM robotics embodiment", "robotics_embodiment"),
    ("audio LLM speech", "audio_speech"),
    ("LLM code generation", "code_generation"),
    ("LLM multi-agent systems", "multi_agent_systems"),
    ("LLM scientific reasoning", "scientific_reasoning")
]

def generate_agent_commands(query, directory):
    """Generate the agent commands for a survey"""
    base_path = f"/path/to/project/output/{directory}/output"
    
    commands = {
        "search": f"""
Search for papers on "{query}".
Use real APIs from /path/to/project/scripts/core/api_clients.py
Generate 20-30 diverse search queries.
Target: 50-100 papers from 2021-2025.
Save to: {base_path}/papers.json and {base_path}/search_report.md
""",
        "cluster": f"""
Cluster papers from {base_path}/papers.json.
Use GPU, real embeddings (all-MiniLM-L6-v2).
Find optimal clusters (5-10 expected).
Save to: {base_path}/clusters.json and {base_path}/cluster_report.md
""",
        "write": f"""
Generate survey from {base_path}/papers.json and {base_path}/clusters.json.
Cite >50% of papers, synthesize findings.
Target: 6000-7000 words.
Save to: {base_path}/survey.md
""",
        "evaluate": f"""
Evaluate survey from {base_path}/survey.md.
Check against {base_path}/papers.json and {base_path}/clusters.json.
Save to: {base_path}/evaluation.json
"""
    }
    return commands

def create_status_tracker():
    """Create a status tracking file"""
    status = {
        "start_time": datetime.now().isoformat(),
        "total_surveys": len(REMAINING_SURVEYS),
        "surveys": []
    }
    
    for query, directory in REMAINING_SURVEYS:
        status["surveys"].append({
            "query": query,
            "directory": directory,
            "status": "pending",
            "commands": generate_agent_commands(query, directory)
        })
    
    status_file = Path("/path/to/project/output/remaining_surveys_status.json")
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    
    print(f"Status tracker created: {status_file}")
    return status

def main():
    print("="*80)
    print("QUICK SURVEY RUNNER")
    print(f"Preparing {len(REMAINING_SURVEYS)} remaining surveys")
    print("="*80)
    
    # Create status tracker
    status = create_status_tracker()
    
    print("\n📋 Remaining Surveys to Process:")
    for i, (query, directory) in enumerate(REMAINING_SURVEYS, 1):
        print(f"{i:2}. {query:<35} -> /output/{directory}/")
    
    print("\n🎯 Agent Commands Generated for Each Survey:")
    print("  1. Paper Search (paper-search-specialist)")
    print("  2. Clustering (topic-mining-clustering)")
    print("  3. Survey Writing (academic-survey-writer)")
    print("  4. Quality Evaluation (survey-quality-evaluator)")
    
    print("\n✨ Ready to process remaining surveys!")
    print("\nTo run each survey, use the Task tool with the appropriate agent and commands.")
    
    return status

if __name__ == "__main__":
    main()