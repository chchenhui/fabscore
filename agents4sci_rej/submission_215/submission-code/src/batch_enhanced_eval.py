#!/usr/bin/env python3
"""
Batch runner for enhanced agent-based evaluation of all completed surveys
"""

import json
import os
from pathlib import Path
from datetime import datetime

# List of completed surveys to evaluate
COMPLETED_SURVEYS = [
    "llm_agent",
    "llm_rlhf_alignment", 
    "instruction_tuning",
    "in_context_learning",
    "synthetic_data",
    "multimodal_llm_rl"
]

def create_evaluation_prompt(survey_name):
    """Create the evaluation prompt for each survey"""
    base_path = f"/path/to/project/output/{survey_name}/output"
    
    prompt = f"""You are an Enhanced Survey Quality Evaluator v3.0. Perform a comprehensive 12-dimensional evaluation using nuanced, context-aware assessment.

Files to evaluate:
- Survey: {base_path}/survey.md
- Papers: {base_path}/papers.json
- Clusters: {base_path}/clusters.json

## Evaluate across 12 dimensions:

### Core Quality (60% weight):
1. Citation Coverage (15%) - Count actual citations vs available papers
2. Accuracy (15%) - Verify claims and attribution
3. Synthesis Quality (15%) - Integration vs listing
4. Organization (15%) - Logical structure and flow

### Writing Quality (20% weight):
5. Readability (5%) - Clarity for target audience
6. Academic Rigor (5%) - Scholarly standards
7. Clarity (5%) - Complex concepts explained well
8. Coherence (5%) - Narrative flow

### Content Depth (20% weight):
9. Comprehensiveness (5%) - Topic coverage
10. Critical Analysis (5%) - Limitations and debates
11. Novelty & Insights (5%) - New perspectives
12. Future Directions (5%) - Research directions

For each dimension:
- Score (0-10) with detailed justification
- Specific examples from text
- Quantitative metrics where applicable

Compare against:
- ACM Computing Surveys: 10,000+ words, >100 citations
- Conference surveys: 6,000-8,000 words, >50 citations
- Workshop papers: 4,000-6,000 words, >30 citations

Output to: {base_path}/enhanced_evaluation_v3.json

Be thorough, fair, and constructive."""
    
    return prompt

def check_survey_files(survey_name):
    """Check if all required files exist for a survey"""
    base_path = Path(f"/path/to/project/output/{survey_name}/output")
    
    required_files = {
        'survey': base_path / 'survey.md',
        'papers': base_path / 'papers.json',
        'clusters': base_path / 'clusters.json'
    }
    
    for file_type, file_path in required_files.items():
        if not file_path.exists():
            return False, f"Missing {file_type} file"
    
    return True, "All files present"

def create_summary_report():
    """Create a summary report of all evaluations"""
    summary = {
        "evaluation_date": datetime.now().isoformat(),
        "evaluator_version": "3.0_agent_based",
        "surveys_evaluated": [],
        "average_scores": {},
        "dimension_averages": {}
    }
    
    all_scores = []
    dimension_scores = {i: [] for i in range(1, 13)}
    
    for survey in COMPLETED_SURVEYS:
        eval_file = Path(f"/path/to/project/output/{survey}/output/enhanced_evaluation_v3.json")
        
        if eval_file.exists():
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                
                overall_score = eval_data.get('overall_assessment', {}).get('score', 0)
                all_scores.append(overall_score)
                
                summary["surveys_evaluated"].append({
                    "survey": survey,
                    "score": overall_score,
                    "grade": eval_data.get('overall_assessment', {}).get('grade', 'N/A'),
                    "publication_ready": eval_data.get('publication_assessment', {}).get('ready_for_submission', False)
                })
                
                # Collect dimension scores
                if 'dimensional_scores' in eval_data:
                    for dim_name, dim_data in eval_data['dimensional_scores'].items():
                        # Map dimension names to numbers for consistency
                        dim_mapping = {
                            'citation_coverage': 1,
                            'accuracy': 2,
                            'synthesis_quality': 3,
                            'organization': 4,
                            'readability': 5,
                            'academic_rigor': 6,
                            'clarity': 7,
                            'coherence': 8,
                            'comprehensiveness': 9,
                            'critical_analysis': 10,
                            'novelty_insights': 11,
                            'future_directions': 12
                        }
                        if dim_name in dim_mapping:
                            dim_num = dim_mapping[dim_name]
                            dimension_scores[dim_num].append(dim_data.get('score', 0))
                
            except Exception as e:
                print(f"Error reading evaluation for {survey}: {e}")
    
    # Calculate averages
    if all_scores:
        summary["average_scores"]["overall"] = round(sum(all_scores) / len(all_scores), 2)
    
    # Calculate dimension averages
    dim_names = [
        "Citation Coverage", "Accuracy", "Synthesis Quality", "Organization",
        "Readability", "Academic Rigor", "Clarity", "Coherence",
        "Comprehensiveness", "Critical Analysis", "Novelty & Insights", "Future Directions"
    ]
    
    for i, name in enumerate(dim_names, 1):
        if dimension_scores[i]:
            summary["dimension_averages"][name] = round(sum(dimension_scores[i]) / len(dimension_scores[i]), 2)
    
    # Save summary
    summary_file = Path("/path/to/project/output/enhanced_evaluation_summary_v3.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    print("=" * 80)
    print("ENHANCED SURVEY EVALUATION BATCH RUNNER v3.0")
    print("Using agent-based evaluation with 12 dimensions")
    print("=" * 80)
    
    results = []
    
    for survey in COMPLETED_SURVEYS:
        print(f"\n📋 Checking {survey}...")
        
        # Check if files exist
        files_ok, message = check_survey_files(survey)
        if not files_ok:
            print(f"  ⚠️ Skipping: {message}")
            continue
        
        # Check if already evaluated
        eval_file = Path(f"/path/to/project/output/{survey}/output/enhanced_evaluation_v3.json")
        if eval_file.exists():
            print(f"  ✅ Already evaluated - loading results")
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                score = eval_data.get('overall_assessment', {}).get('score', 'N/A')
                grade = eval_data.get('overall_assessment', {}).get('grade', 'N/A')
                print(f"  📊 Score: {score}/10 (Grade: {grade})")
                results.append((survey, score, grade))
            except:
                print(f"  ❌ Error reading evaluation")
        else:
            print(f"  🔄 Needs evaluation - creating prompt")
            prompt = create_evaluation_prompt(survey)
            
            # Save prompt for manual execution
            prompt_file = Path(f"/path/to/project/output/{survey}/output/eval_prompt.txt")
            with open(prompt_file, 'w') as f:
                f.write(prompt)
            print(f"  💾 Prompt saved to: {prompt_file}")
            print(f"  ⚡ Run evaluation using survey-quality-evaluator agent")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if results:
        print("\n📊 Evaluated Surveys:")
        for survey, score, grade in results:
            print(f"  - {survey:25} Score: {score}/10 (Grade: {grade})")
        
        avg_score = sum(s for _, s, _ in results if isinstance(s, (int, float))) / len(results)
        print(f"\n📈 Average Score: {avg_score:.2f}/10")
    
    # Create summary report
    print("\n📁 Creating summary report...")
    summary = create_summary_report()
    print(f"✅ Summary saved to: enhanced_evaluation_summary_v3.json")
    
    print("\n✨ Evaluation batch processing complete!")

if __name__ == '__main__':
    main()