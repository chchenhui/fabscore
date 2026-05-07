#!/usr/bin/env python3
import json
import os

# AutoSurvey evaluation files (from evaluation_results folder)
autosurvey_files = {
    "In-Context Learning": "/path/to/project/evaluation_results/In-Context_Learning_evaluation.json",
    "Instruction Tuning": "/path/to/project/evaluation_results/Instruction_Tuning_evaluation.json",
    "RLHF Alignment": "/path/to/project/evaluation_results/RLHF_Alignment_evaluation.json",
    "Synthetic Data": "/path/to/project/evaluation_results/Synthetic_Data_evaluation.json",
    "Multimodal LLM RL": "/path/to/project/evaluation_results/Multimodal_LLM_RL_evaluation.json"
}

results = {}

for topic, filepath in autosurvey_files.items():
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract category scores
        cat_scores = data.get('category_scores', {})
        
        results[topic] = {
            'overall': data.get('overall_assessment', {}).get('overall_score', 0),
            'core_quality': cat_scores.get('core_quality', {}).get('score', 0),
            'writing_quality': cat_scores.get('writing_quality', {}).get('score', 0),
            'content_depth': cat_scores.get('content_depth', {}).get('score', 0)
        }
        
        print(f"{topic}:")
        print(f"  Overall: {results[topic]['overall']}")
        print(f"  Core Quality: {results[topic]['core_quality']}")
        print(f"  Writing Quality: {results[topic]['writing_quality']}")
        print(f"  Content Depth: {results[topic]['content_depth']}")
        print()

# Calculate averages
if results:
    avg_overall = sum(r['overall'] for r in results.values()) / len(results)
    avg_core = sum(r['core_quality'] for r in results.values()) / len(results)
    avg_writing = sum(r['writing_quality'] for r in results.values()) / len(results)
    avg_content = sum(r['content_depth'] for r in results.values()) / len(results)
    
    print("Averages:")
    print(f"  Overall: {avg_overall:.2f}")
    print(f"  Core Quality: {avg_core:.2f}")
    print(f"  Writing Quality: {avg_writing:.2f}")
    print(f"  Content Depth: {avg_content:.2f}")

# Save to JSON
with open("/path/to/project/autosurvey_category_scores.json", "w") as f:
    json.dump(results, f, indent=2)