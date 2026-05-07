#!/usr/bin/env python3
"""
Convert journal.json files to summary files needed for writeup generation.
This is a simplified conversion for when summary files are missing.
"""

import json
import os

def convert_journals_to_summaries():
    base_path = "logs/0-run"

    # Define stage mappings
    stage_mappings = {
        "stage_1_initial_implementation_1_preliminary": "baseline_summary.json",
        "stage_2_baseline_tuning_1_first_attempt": "research_summary.json",
        "stage_3_creative_research_1_first_attempt": "research_summary.json",
        "stage_4_ablation_studies_1_first_attempt": "ablation_summary.json"
    }

    summaries = {}

    for stage_dir, summary_name in stage_mappings.items():
        journal_path = os.path.join(base_path, stage_dir, "journal.json")
        if os.path.exists(journal_path):
            try:
                with open(journal_path, 'r') as f:
                    journal_data = json.load(f)

                # Extract key information from journal
                summary = {
                    "best node": {
                        "overall_plan": journal_data.get("overall_plan", "No plan available"),
                        "analysis": journal_data.get("analysis", "No analysis available"),
                        "metric": journal_data.get("final_metric", journal_data.get("metric", {})),
                        "vlm_feedback_summary": "Generated from experiment results"
                    }
                }

                summaries[summary_name] = summary
                print(f"Created summary for {stage_dir} -> {summary_name}")

            except Exception as e:
                print(f"Error processing {journal_path}: {e}")

    # Save summaries
    for summary_name, summary_data in summaries.items():
        summary_path = os.path.join(base_path, summary_name)
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved {summary_path}")

    # Create ablation summary if we have multiple stages
    if len(summaries) > 1:
        ablation_data = []
        for i, (summary_name, summary_data) in enumerate(summaries.items()):
            ablation_entry = {
                "ablation_name": f"experiment_{i+1}",
                **summary_data.get("best node", {})
            }
            ablation_data.append(ablation_entry)

        ablation_path = os.path.join(base_path, "ablation_summary.json")
        with open(ablation_path, 'w') as f:
            json.dump(ablation_data, f, indent=2)
        print(f"Saved {ablation_path}")

if __name__ == "__main__":
    convert_journals_to_summaries()