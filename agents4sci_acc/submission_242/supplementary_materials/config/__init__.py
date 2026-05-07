"""
Configuration management for the AI Agent Marketplace Research Platform.
"""

import os
from pathlib import Path

# Try to import actual config, fall back to defaults
try:
    from .config import *
except ImportError:
    # Default configuration if config.py doesn't exist
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = "gpt-5-mini"
    DEFAULT_ROUNDS = 50
    DEFAULT_NUM_FREELANCERS = 20
    DEFAULT_NUM_CLIENTS = 10
    RESULTS_DIR = "results"
    FIGURES_DIR = "results/figures"
    LOG_LEVEL = "INFO"
    LOG_TO_FILE = True

# Ensure results directories exist
Path(RESULTS_DIR).mkdir(exist_ok=True)
Path(FIGURES_DIR).mkdir(exist_ok=True)
