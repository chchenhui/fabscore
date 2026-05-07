#!/usr/bin/env python3
"""
Batch Survey Generator - Efficiently processes multiple survey queries
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

# Survey queries configuration
SURVEYS = [
    {
        "query": "instruction tuning LLM",
        "dir": "instruction_tuning",
        "status": "pending"
    },
    {
        "query": "in-context learning alignment",
        "dir": "in_context_learning",
        "status": "pending"
    },
    {
        "query": "synthetic data LLM training",
        "dir": "synthetic_data",
        "status": "pending"
    },
    {
        "query": "LLM data curation",
        "dir": "data_curation",
        "status": "pending"
    },
    {
        "query": "LLM pretraining datasets",
        "dir": "pretraining_datasets",
        "status": "pending"
    },
    {
        "query": "LLM evaluation benchmarks",
        "dir": "evaluation_benchmarks",
        "status": "pending"
    },
    {
        "query": "LLM safety jailbreaking",
        "dir": "safety_jailbreaking",
        "status": "pending"
    },
    {
        "query": "LLM bias fairness",
        "dir": "bias_fairness",
        "status": "pending"
    },
    {
        "query": "LLM quantization compression",
        "dir": "quantization_compression",
        "status": "pending"
    },
    {
        "query": "LLM scaling laws",
        "dir": "scaling_laws",
        "status": "pending"
    },
    {
        "query": "efficient LLM inference",
        "dir": "efficient_inference",
        "status": "pending"
    },
    {
        "query": "multimodal LLM vision",
        "dir": "multimodal_vision",
        "status": "pending"
    },
    {
        "query": "LLM robotics embodiment",
        "dir": "robotics_embodiment",
        "status": "pending"
    },
    {
        "query": "audio LLM speech",
        "dir": "audio_speech",
        "status": "pending"
    },
    {
        "query": "LLM code generation",
        "dir": "code_generation",
        "status": "pending"
    },
    {
        "query": "LLM multi-agent systems",
        "dir": "multi_agent_systems",
        "status": "pending"
    },
    {
        "query": "LLM scientific reasoning",
        "dir": "scientific_reasoning",
        "status": "pending"
    }
]

def create_survey_script(survey_info):
    """Create a shell script to run the survey pipeline for a single query"""
    script_content = f"""#!/bin/bash
# Survey Generation Script for: {survey_info['query']}
# Generated: {datetime.now().isoformat()}

QUERY="{survey_info['query']}"
OUTPUT_DIR="/path/to/project/output/{survey_info['dir']}"

echo "=========================================="
echo "Starting Survey Generation for: $QUERY"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR/output"

# Record start time
echo "{{" > "$OUTPUT_DIR/output/pipeline_status.json"
echo '  "query": "'"$QUERY"'",' >> "$OUTPUT_DIR/output/pipeline_status.json"
echo '  "start_time": "'$(date -Iseconds)'",' >> "$OUTPUT_DIR/output/pipeline_status.json"
echo '  "status": "running"' >> "$OUTPUT_DIR/output/pipeline_status.json"
echo "}}" >> "$OUTPUT_DIR/output/pipeline_status.json"

echo "Pipeline initialized for $QUERY"
echo "Ready for agent execution..."
"""
    
    script_path = Path(f"/path/to/project/output/{survey_info['dir']}/run_pipeline.sh")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    return script_path

def prepare_all_surveys():
    """Prepare all survey directories and scripts"""
    print("="*80)
    print("BATCH SURVEY PREPARATION")
    print(f"Preparing {len(SURVEYS)} survey pipelines")
    print("="*80)
    
    summary = {
        "total_surveys": len(SURVEYS),
        "prepared_at": datetime.now().isoformat(),
        "surveys": []
    }
    
    for i, survey in enumerate(SURVEYS, 1):
        print(f"\n[{i}/{len(SURVEYS)}] Preparing: {survey['query']}")
        
        # Create directory structure
        output_dir = Path(f"/path/to/project/output/{survey['dir']}/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run script
        script_path = create_survey_script(survey)
        
        # Create initial status file
        status = {
            "query": survey['query'],
            "directory": survey['dir'],
            "status": "prepared",
            "prepared_at": datetime.now().isoformat(),
            "script": str(script_path)
        }
        
        status_file = output_dir / "initial_status.json"
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        summary["surveys"].append(status)
        print(f"  ✓ Directory created: {output_dir}")
        print(f"  ✓ Script created: {script_path}")
        print(f"  ✓ Status file: {status_file}")
    
    # Save overall summary
    summary_file = Path("/path/to/project/output/batch_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("PREPARATION COMPLETE")
    print(f"All {len(SURVEYS)} surveys are ready for agent execution")
    print(f"Summary saved to: {summary_file}")
    print("="*80)
    
    print("\n📋 Survey Directories Created:")
    for survey in SURVEYS:
        print(f"  - /output/{survey['dir']}/")
    
    print("\n✨ Ready to run agents for each survey!")
    
    return summary

if __name__ == "__main__":
    prepare_all_surveys()