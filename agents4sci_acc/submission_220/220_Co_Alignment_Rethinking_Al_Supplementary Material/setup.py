from setuptools import setup, find_packages

setup(
    name="bica",
    version="0.1.0",
    description="Bidirectional Cognitive Adaptation (BiCA) Framework",
    author="BiCA Team",
    packages=find_packages(),
    install_requires=[
        # Core dependencies - most should be installed via conda
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "scikit-learn>=1.1.0",
        # Pip-only dependencies
        "tensorboard>=2.8.0",
        "wandb>=0.13.0",
        "gymnasium>=0.26.0", # Replaced gym
        "statsmodels>=0.13.0",
        "pot",  # Python Optimal Transport
    ],
    python_requires=">=3.9",
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
        ],
        "ui": [
            "streamlit>=1.15.0",
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
        ],
        "optimization": [
            "optuna>=3.0.0",
            "hydra-core>=1.2.0",
        ],
    },
)
