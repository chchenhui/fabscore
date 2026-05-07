#!/usr/bin/env python3
"""
Run experiments for all seed ideas in experiment/cs/seed_ideas.json
Each seed idea will be processed separately using launch.py
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any


def load_seed_ideas(seed_file: str) -> List[Dict[str, Any]]:
    """Load seed ideas from JSON file"""
    with open(seed_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    else:
        # If it's a single object, wrap it in a list
        return [data]


def load_strategies(strategy_file: str) -> Dict[str, str]:
    """Load strategies from JSON file"""
    with open(strategy_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def generate_strategy_combinations(seed_ideas: List[Dict[str, Any]], strategies: Dict[str, str]) -> List[Dict[str, Any]]:
    """Generate all combinations of seeds with strategies"""
    combinations = []
    
    for seed_idea in seed_ideas:
        seed_text = seed_idea.get('seed', '')
        
        # Single strategy combinations (s1, s2, s3, s4, s5)
        for strategy_key, strategy_text in strategies.items():
            combination = {
                'seed': seed_text,
                'strategy_key': strategy_key,
                'strategy_text': strategy_text,
                'combination_type': 'single'
            }
            combinations.append(combination)
        
        # Combined strategy (s1 + s2 + s3 + s4 + s5)
        combined_strategy = '\n\n'.join(strategies.values())
        combination = {
            'seed': seed_text,
            'strategy_key': 'combined',
            'strategy_text': combined_strategy,
            'combination_type': 'combined'
        }
        combinations.append(combination)
    
    return combinations


def create_temp_seed_file(seed_idea: Dict[str, Any], temp_dir: str) -> str:
    """Create a temporary seed file for a single idea with strategy"""
    # Create a unique filename based on content
    content_hash = hash(str(seed_idea))
    temp_file = os.path.join(temp_dir, f"temp_seed_{content_hash}.json")
    
    # Format the seed file with strategy information
    seed_content = {
        'seed': seed_idea['seed']
    }
    
    # Add strategy information if present
    if 'strategy_text' in seed_idea:
        strategy_key = seed_idea.get('strategy_key', 'unknown')
        strategy_text = seed_idea['strategy_text']
        
        # Append strategy to the seed
        seed_content['seed'] += f"\n\n## Strategy: {strategy_key.upper()}\n\n{strategy_text}"
    
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(seed_content, f, indent=2, ensure_ascii=False)
    return temp_file


def run_experiment(
    template_dir: str,
    seed_file: str,
    num_ideas: int,
    out_dir: str,
    enable_review: bool,
    experiment_name: str
) -> bool:
    """Run a single experiment using launch.py"""
    cmd = [
        sys.executable,
        "launch.py",
        "--template-dir", template_dir,
        "--seed-path", seed_file,
        "--num-ideas", str(num_ideas),
        "--out", out_dir
    ]

    if enable_review:
        cmd.append("--enable-review")

    print(f"Running experiment: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per experiment
        )

        if result.returncode == 0:
            print(f"✓ Experiment {experiment_name} completed successfully")
            return True
        else:
            print(f"✗ Experiment {experiment_name} failed")
            print(f"Error output: {result.stderr[:500]}...")  # Truncate long error messages
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ Experiment {experiment_name} timed out after 1 hour")
        return False
    except Exception as e:
        print(f"✗ Experiment {experiment_name} failed with exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run experiments for all seed ideas with strategies")
    parser.add_argument(
        "--seed-file",
        default="experiment/cs/seed_ideas.json",
        help="Path to seed ideas JSON file"
    )
    parser.add_argument(
        "--strategy-file",
        default="experiment/cs/strategy.json",
        help="Path to strategies JSON file"
    )
    parser.add_argument(
        "--template-dir",
        default="experiment/cs/latex",
        help="Template directory path"
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=4,
        help="Number of ideas to generate per seed"
    )
    parser.add_argument(
        "--out-base",
        default="experiment/cs/results",
        help="Base output directory"
    )
    parser.add_argument(
        "--enable-review",
        action="store_true",
        help="Enable LLM review step"
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Maximum number of seeds to process (for testing)"
    )

    args = parser.parse_args()

    # Load seed ideas
    seed_file_path = args.seed_file
    if not os.path.exists(seed_file_path):
        print(f"Error: Seed file not found: {seed_file_path}")
        return 1

    seed_ideas = load_seed_ideas(seed_file_path)
    print(f"Loaded {len(seed_ideas)} seed ideas from {seed_file_path}")

    # Load strategies
    strategy_file_path = args.strategy_file
    if not os.path.exists(strategy_file_path):
        print(f"Error: Strategy file not found: {strategy_file_path}")
        return 1

    strategies = load_strategies(strategy_file_path)
    print(f"Loaded {len(strategies)} strategies from {strategy_file_path}")

    # Create base output directory
    os.makedirs(args.out_base, exist_ok=True)

    # Process seeds sequentially
    total_seeds = min(len(seed_ideas), args.max_seeds) if args.max_seeds else len(seed_ideas)
    successful_experiments = 0
    total_experiments = 0

    for seed_idx, seed_idea in enumerate(seed_ideas[:total_seeds]):
        seed_text = seed_idea.get('seed', '')
        print(f"\n{'='*60}")
        print(f"Processing seed {seed_idx+1}/{total_seeds}: {seed_text[:60]}...")
        print(f"{'='*60}")

        # Generate strategy combinations for this seed
        seed_combinations = []
        for strategy_key, strategy_text in strategies.items():
            combination = {
                'seed': seed_text,
                'strategy_key': strategy_key,
                'strategy_text': strategy_text,
                'combination_type': 'single'
            }
            seed_combinations.append(combination)
        
        # Add combined strategy
        combined_strategy = 'Experimental Requirements: '.join(strategies.values())
        combination = {
            'seed': seed_text,
            'strategy_key': 'combined',
            'strategy_text': combined_strategy,
            'combination_type': 'combined'
        }
        seed_combinations.append(combination)

        print(f"Generated {len(seed_combinations)} strategy combinations for this seed")

        # Create temporary directory for this seed's experiments
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use ThreadPoolExecutor for parallel processing of strategy combinations
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Use exactly the number of strategy combinations as workers (6 workers for 6 strategies)
            num_workers = len(seed_combinations)
            print(f"Using {num_workers} workers for {len(seed_combinations)} strategy combinations")
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all strategy combinations for this seed
                future_to_strategy = {}
                
                for combo_idx, combination in enumerate(seed_combinations):
                    strategy_key = combination['strategy_key']
                    
                    # Create temporary seed file for this combination
                    temp_seed_file = create_temp_seed_file(combination, temp_dir)
                    
                    # Create unique output directory for this strategy combination
                    # Format: {seed_idx}_{strategy_key} (e.g., 001_S1, 001_all)
                    if strategy_key == 'combined':
                        strategy_dir_name = f"{seed_idx+1:03d}_all"
                    else:
                        strategy_dir_name = f"{seed_idx+1:03d}_{strategy_key}"
                    
                    out_dir = os.path.join(args.out_base, strategy_dir_name)
                    
                    print(f"Submitting strategy {strategy_key} -> output dir: {strategy_dir_name} (combination {combo_idx+1}/{len(seed_combinations)})")
                    
                    # Submit the strategy combination (which will run num_ideas experiments internally)
                    future = executor.submit(
                        run_strategy_combination,
                        args.template_dir,
                        temp_seed_file,
                        args.num_ideas,
                        out_dir,
                        args.enable_review,
                        strategy_dir_name
                    )
                    future_to_strategy[future] = (strategy_dir_name, combination)

                # Process completed strategy combinations
                seed_successful = 0
                seed_total = len(seed_combinations)
                completed_count = 0
                
                for future in as_completed(future_to_strategy):
                    experiment_name, combination = future_to_strategy[future]
                    completed_count += 1
                    strategy_key = combination['strategy_key']
                    
                    print(f"Processing result for strategy {strategy_key} ({completed_count}/{seed_total})")
                    
                    try:
                        success = future.result()
                        if success > 0:
                            seed_successful += success
                            successful_experiments += success
                            total_experiments += args.num_ideas  # Each strategy combination runs num_ideas experiments
                            
                            print(f"✓ Strategy {strategy_key}: {success}/{args.num_ideas} experiments successful")
                        else:
                            print(f"✗ Strategy {strategy_key}: 0/{args.num_ideas} experiments successful")
                            total_experiments += args.num_ideas
                        
                    except Exception as e:
                        print(f"✗ Strategy {strategy_key} failed with exception: {e}")
                        total_experiments += args.num_ideas

                print(f"Seed {seed_idx+1} summary: {seed_successful}/{seed_total * args.num_ideas} experiments successful")

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY:")
    print(f"Total seeds processed: {total_seeds}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful experiments: {successful_experiments}")
    print(f"Failed experiments: {total_experiments - successful_experiments}")
    print(f"Success rate: {successful_experiments/total_experiments*100:.1f}%" if total_experiments > 0 else "Success rate: N/A")
    print(f"{'='*60}")

    return 0 if successful_experiments > 0 else 1


def run_strategy_combination(template_dir, seed_file, num_ideas, out_dir, enable_review, experiment_base_name):
    """Run a single strategy combination by calling launch.py once with num_ideas"""
    print(f"Running strategy combination: {experiment_base_name} with {num_ideas} ideas")
    
    try:
        # Create output directory
        os.makedirs(out_dir, exist_ok=True)
        
        # Build command - call launch.py once with the desired number of ideas
        cmd = [
            "python", "launch.py",
            "--template-dir", template_dir,
            "--seed-path", seed_file,
            "--num-ideas", str(num_ideas),
            "--out", out_dir
        ]
        
        if enable_review:
            cmd.append("--enable-review")
            print(f"  Command: {' '.join(cmd)} (with review enabled)")
        else:
            print(f"  Command: {' '.join(cmd)} (review disabled)")
        
        # Run the experiment
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print(f"  ✓ Strategy {experiment_base_name} completed successfully")
            if result.stdout:
                print(f"  Output: {result.stdout.strip()}")
            return num_ideas  # All ideas were successful
        else:
            print(f"  ✗ Strategy {experiment_base_name} failed with return code {result.returncode}")
            if result.stderr:
                print(f"  Error output: {result.stderr[:300]}...")
            if result.stdout:
                print(f"  Standard output: {result.stdout[:300]}...")
            return 0  # No ideas were successful
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Strategy {experiment_base_name} timed out after 1 hour")
        return 0
    except Exception as e:
        print(f"  ✗ Strategy {experiment_base_name} failed with exception: {e}")
        return 0


if __name__ == "__main__":
    sys.exit(main())