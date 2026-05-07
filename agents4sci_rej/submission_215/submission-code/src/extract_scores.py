#!/usr/bin/env python3
import json
import os
from pathlib import Path

# Define paths for each system
multi_agent_paths = {
    "In-Context Learning": "/path/to/project/output/in_context_learning/output/enhanced_evaluation_v3.json",
    "Instruction Tuning": "/path/to/project/output/instruction_tuning/output/enhanced_evaluation_v3.json",
    "LLM Agents": "/path/to/project/output/llm_agent/output/enhanced_evaluation_v3.json",
    "RLHF Alignment": "/path/to/project/output/llm_rlhf_alignment/output/enhanced_evaluation_v3.json",
    "Synthetic Data": "/path/to/project/output/synthetic_data/output/enhanced_evaluation_v3.json",
    "Multimodal LLM RL": "/path/to/project/output/multimodal_llm_rl/output/enhanced_evaluation_v3.json"
}

# AutoSurvey scores from the comprehensive summary (evaluation_results folder)
autosurvey_scores = {
    "In-Context Learning": {
        "citation_coverage": 3.5,
        "accuracy": 5.5,
        "synthesis_quality": 3.0,
        "organization": 6.0,
        "readability": 5.5,
        "academic_rigor": 4.5,
        "clarity": 5.0,
        "coherence": 5.0,
        "comprehensiveness": 5.5,
        "critical_analysis": 3.5,
        "novelty_insights": 4.0,
        "future_directions": 6.5,
        "overall": 4.8
    },
    "Instruction Tuning": {
        "citation_coverage": 4.0,
        "accuracy": 4.5,
        "synthesis_quality": 2.5,
        "organization": 5.0,
        "readability": 3.5,
        "academic_rigor": 3.0,
        "clarity": 4.0,
        "coherence": 4.5,
        "comprehensiveness": 4.5,
        "critical_analysis": 3.0,
        "novelty_insights": 3.5,
        "future_directions": 7.0,
        "overall": 4.2
    },
    "RLHF Alignment": {
        "citation_coverage": 4.5,
        "accuracy": 7.0,
        "synthesis_quality": 4.5,
        "organization": 7.0,
        "readability": 6.5,
        "academic_rigor": 6.0,
        "clarity": 6.5,
        "coherence": 7.0,
        "comprehensiveness": 7.0,
        "critical_analysis": 4.5,
        "novelty_insights": 5.0,
        "future_directions": 7.0,
        "overall": 6.2
    },
    "Synthetic Data": {
        "citation_coverage": 5.0,
        "accuracy": 6.0,
        "synthesis_quality": 3.5,
        "organization": 6.5,
        "readability": 6.0,
        "academic_rigor": 5.5,
        "clarity": 6.0,
        "coherence": 6.0,
        "comprehensiveness": 6.5,
        "critical_analysis": 4.0,
        "novelty_insights": 4.5,
        "future_directions": 7.0,
        "overall": 5.8
    },
    "Multimodal LLM RL": {
        "citation_coverage": 3.0,
        "accuracy": 4.0,
        "synthesis_quality": 2.0,
        "organization": 3.5,
        "readability": 2.5,
        "academic_rigor": 4.0,
        "clarity": 3.0,
        "coherence": 3.0,
        "comprehensiveness": 5.0,
        "critical_analysis": 2.5,
        "novelty_insights": 3.0,
        "future_directions": 7.0,
        "overall": 3.8
    }
}

# Extract multi-agent scores
multi_agent_scores = {}

for topic, path in multi_agent_paths.items():
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            
        scores = {}
        
        # Extract dimensional scores
        if 'dimensional_scores' in data:
            dims = data['dimensional_scores']
            
            # Core quality
            if 'core_quality' in dims:
                scores['citation_coverage'] = dims['core_quality'].get('citation_coverage', {}).get('score', 0)
                scores['accuracy'] = dims['core_quality'].get('accuracy', {}).get('score', 0)
                scores['synthesis_quality'] = dims['core_quality'].get('synthesis_quality', {}).get('score', 0)
                scores['organization'] = dims['core_quality'].get('organization', {}).get('score', 0)
            
            # Writing quality
            if 'writing_quality' in dims:
                scores['readability'] = dims['writing_quality'].get('readability', {}).get('score', 0)
                scores['academic_rigor'] = dims['writing_quality'].get('academic_rigor', {}).get('score', 0)
                scores['clarity'] = dims['writing_quality'].get('clarity', {}).get('score', 0)
                scores['coherence'] = dims['writing_quality'].get('coherence', {}).get('score', 0)
            
            # Content depth
            if 'content_depth' in dims:
                scores['comprehensiveness'] = dims['content_depth'].get('comprehensiveness', {}).get('score', 0)
                scores['critical_analysis'] = dims['content_depth'].get('critical_analysis', {}).get('score', 0)
                scores['novelty_insights'] = dims['content_depth'].get('novelty_insights', {}).get('score', 0)
                scores['future_directions'] = dims['content_depth'].get('future_directions', {}).get('score', 0)
        
        # Get overall score
        scores['overall'] = data.get('overall_quality_score', {}).get('final_score', 0)
        
        multi_agent_scores[topic] = scores
    else:
        print(f"File not found: {path}")

# Print results as LaTeX table
print("% Multi-Agent Scores")
for topic in multi_agent_scores:
    scores = multi_agent_scores[topic]
    print(f"% {topic}: {scores.get('overall', 'N/A')}")

print("\n% AutoSurvey Scores")  
for topic in autosurvey_scores:
    scores = autosurvey_scores[topic]
    print(f"% {topic}: {scores.get('overall', 'N/A')}")

# Save to JSON for reference
output = {
    "multi_agent": multi_agent_scores,
    "autosurvey": autosurvey_scores
}

with open("/path/to/project/all_system_scores.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nScores saved to all_system_scores.json")