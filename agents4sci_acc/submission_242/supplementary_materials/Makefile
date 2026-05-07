# AI Agent Marketplace Research Platform Makefile

.PHONY: help install test run clean setup-dev

help:
	@echo "AI Agent Marketplace Research Platform"
	@echo "======================================"
	@echo ""
	@echo "Available commands:"
	@echo "  install        Install dependencies"
	@echo "  setup-dev      Set up development environment"
	@echo "  run           Run marketplace simulation"
	@echo "  test           Run all tests"
	@echo "  clean          Clean generated files"
	@echo "  compile-paper  Compile LaTeX research paper"
	@echo ""

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Installation complete!"

setup-dev: install
	@echo "Setting up development environment..."
	pip install -e .
	@echo "Development environment ready!"

run:
	@echo "Running marketplace simulation..."
	python run_marketplace.py
	@echo "Simulation complete!"

test:
	@echo "Running tests..."
	python -m pytest tests/
	@echo "All tests passed!"

clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	@echo "Cleanup complete!"

compile-paper:
	@echo "Compiling research paper..."
	cd papers && pdflatex comprehensive_research_paper.tex