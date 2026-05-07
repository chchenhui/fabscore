"""
Setup script for AI Agent Marketplace Research Platform
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-agent-marketplace",
    version="1.0.0",
    author="Anonymous",
    author_email="[CONTACT_EMAIL]",
    description="A comprehensive research platform for studying emergent economic behavior in AI agent societies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="[REPOSITORY_URL]",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=3.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-marketplace=src.simulation.mock_simulation:main",
            "ai-analysis=src.analysis.comprehensive_analysis:main",
        ],
    },
    project_urls={
        "Bug Reports": "[REPOSITORY_URL]/issues",
        "Source": "[REPOSITORY_URL]",
        "Documentation": "[REPOSITORY_URL]/blob/main/docs/",
    },
)
