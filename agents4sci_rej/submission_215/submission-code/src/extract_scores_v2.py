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

# Extract multi-agent scores
multi_agent_scores = {}

for topic, path in multi_agent_paths.items():
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            
        scores = {}
        
        # Extract scores from category_scores structure
        if 'category_scores' in data:
            cats = data['category_scores']
            
            # Core quality
            if 'core_quality' in cats and 'breakdown' in cats['core_quality']:
                core = cats['core_quality']['breakdown']
                scores['citation_coverage'] = core.get('citation_coverage', 0)
                scores['accuracy'] = core.get('accuracy', 0)
                scores['synthesis_quality'] = core.get('synthesis_quality', 0)
                scores['organization'] = core.get('organization', 0)
            
            # Writing quality
            if 'writing_quality' in cats and 'breakdown' in cats['writing_quality']:
                writing = cats['writing_quality']['breakdown']
                scores['readability'] = writing.get('readability', 0)
                scores['academic_rigor'] = writing.get('academic_rigor', 0)
                scores['clarity'] = writing.get('clarity', 0)
                scores['coherence'] = writing.get('coherence', 0)
            
            # Content depth
            if 'content_depth' in cats and 'breakdown' in cats['content_depth']:
                content = cats['content_depth']['breakdown']
                scores['comprehensiveness'] = content.get('comprehensiveness', 0)
                scores['critical_analysis'] = content.get('critical_analysis', 0)
                scores['novelty_insights'] = content.get('novelty_insights', 0)
                scores['future_directions'] = content.get('future_directions', 0)
        
        # Get overall score
        if 'overall_assessment' in data:
            scores['overall'] = data['overall_assessment'].get('weighted_score', 0)
        
        multi_agent_scores[topic] = scores
        print(f"Extracted {topic}: Overall={scores.get('overall', 'N/A')}")
    else:
        print(f"File not found: {path}")

# Save to JSON
with open("/path/to/project/multi_agent_scores.json", "w") as f:
    json.dump(multi_agent_scores, f, indent=2)

print(f"\nExtracted scores for {len(multi_agent_scores)} topics")