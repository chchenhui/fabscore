#!/usr/bin/env python3
"""
Run AutoSurvey with local OpenAI-compatible API (e.g., vLLM)
Generates surveys for the 6 test queries from the paper
"""

import os
import sys
import argparse
from main import main as autosurvey_main

# The 6 input queries from the evaluation table
TEST_QUERIES = [
    "Instruction Tuning",           # Papers: 100, Citation: 99%, Grade: A-
    "LLM Agents",                    # Papers: 100, Citation: 79%, Grade: B+
    "RLHF Alignment",                # Papers: 443, Citation: 10%, Grade: B
    "Synthetic Data",                # Papers: 100, Citation: 47%, Grade: B
    "In-Context Learning",           # Papers: 100, Citation: 28%, Grade: C+
    "Multimodal LLM RL"              # Papers: 75,  Citation: 12%, Grade: C+
]

def run_survey_with_local_api(
    topic,
    local_api_url="http://localhost:8000/v1/chat/completions",
    model_name="meta-llama/Llama-3.1-8B-Instruct",  # or your model
    api_key="dummy",  # vLLM doesn't need real key
    output_dir="./local_output",
    db_path="./database"
):
    """
    Run AutoSurvey with local OpenAI-compatible API
    
    Args:
        topic: Research topic to generate survey for
        local_api_url: Your local vLLM or other OpenAI-compatible endpoint
        model_name: Model name as served by your local instance
        api_key: API key (can be dummy for local instances)
        output_dir: Directory to save generated surveys
        db_path: Path to AutoSurvey database
    """
    
    # Create namespace object to mimic argparse args
    class Args:
        pass
    
    args = Args()
    args.gpu = '0'
    args.saving_path = output_dir
    args.model = model_name
    args.topic = topic
    args.section_num = 7
    args.subsection_len = 700
    args.outline_reference_num = 1500
    args.rag_num = 60
    args.api_url = local_api_url
    args.api_key = api_key
    args.db_path = db_path
    args.embedding_model = 'nomic-ai/nomic-embed-text-v1'
    
    print(f"\n{'='*60}")
    print(f"Generating survey for: {topic}")
    print(f"Using API: {local_api_url}")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")
    
    try:
        autosurvey_main(args)
        print(f"✓ Survey generated successfully for: {topic}")
        print(f"  Saved to: {output_dir}/{topic}.md")
    except Exception as e:
        print(f"✗ Error generating survey for {topic}: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run AutoSurvey with local OpenAI-compatible API')
    parser.add_argument('--api-url', default='http://localhost:8000/v1/chat/completions',
                        help='Local API endpoint URL')
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Model name as served by your local instance')
    parser.add_argument('--api-key', default='dummy',
                        help='API key (can be dummy for local instances)')
    parser.add_argument('--output-dir', default='./local_output',
                        help='Directory to save generated surveys')
    parser.add_argument('--db-path', default='./database',
                        help='Path to AutoSurvey database')
    parser.add_argument('--query', type=int, choices=range(1, 7),
                        help='Run specific query (1-6), or omit to run all')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select queries to run
    if args.query:
        queries = [TEST_QUERIES[args.query - 1]]
        print(f"Running single query: {queries[0]}")
    else:
        queries = TEST_QUERIES
        print(f"Running all 6 test queries")
    
    # Run surveys
    success_count = 0
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Processing: {query}")
        if run_survey_with_local_api(
            topic=query,
            local_api_url=args.api_url,
            model_name=args.model,
            api_key=args.api_key,
            output_dir=args.output_dir,
            db_path=args.db_path
        ):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(queries)} surveys generated successfully")
    print(f"Output saved to: {args.output_dir}/")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()