"""
Configuration file template.
Copy this file to config.py and fill in your actual values.
"""

# OpenAI API Configuration (only needed for live LLM simulation)
OPENAI_API_KEY = "your-openai-api-key-here"  # Set this if using real LLM calls
OPENAI_MODEL = "gpt-4o-mini"  # or "gpt-3.5-turbo"

# Simulation Configuration
DEFAULT_ROUNDS = 50
DEFAULT_NUM_FREELANCERS = 20
DEFAULT_NUM_CLIENTS = 10

# Output Configuration
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_TO_FILE = True
