#!/usr/bin/env python3
"""
Main entry point for the GPT-powered marketplace simulation.
Run experiments with different relevance algorithms and configurations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import the main marketplace class
from marketplace.true_gpt_marketplace import main

if __name__ == "__main__":
    main()
