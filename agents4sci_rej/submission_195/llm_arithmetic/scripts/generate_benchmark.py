#!/usr/bin/env python3
"""Generate the MATH 401+ benchmark dataset."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_arithmetic.evaluation.benchmark_generator import BenchmarkGenerator


def main():
    # Setup directories
    data_dir = Path(__file__).parent.parent / "data" / "benchmarks"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate benchmark
    generator = BenchmarkGenerator(seed=42)
    problems = generator.generate_math401_benchmark()
    
    # Save benchmark
    benchmark_file = data_dir / "math401_plus.json"
    generator.save_benchmark(problems, str(benchmark_file))
    
    # Print statistics
    operations = {}
    difficulties = {}
    
    for problem in problems:
        operations[problem.operation] = operations.get(problem.operation, 0) + 1
        difficulties[problem.difficulty] = difficulties.get(problem.difficulty, 0) + 1
    
    print(f"\nBenchmark Statistics:")
    print(f"Total problems: {len(problems)}")
    print(f"\nBy operation:")
    for op, count in sorted(operations.items()):
        print(f"  {op}: {count}")
    print(f"\nBy difficulty:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff}: {count}")


if __name__ == "__main__":
    main()