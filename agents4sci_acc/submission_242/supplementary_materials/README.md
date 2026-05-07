# 🤖 Simulated Marketplace - LLM-Powered Freelance Market Simulation

[![Research Status](https://img.shields.io/badge/Research-Active-orange)](docs/project_description.md)
[![Implementation](https://img.shields.io/badge/Implementation-Complete-green)](#quick-start)
[![API](https://img.shields.io/badge/API-OpenAI_GPT4-blue)](#prerequisites)

> **A reproducible framework for studying emergent economic behavior with LLM agents in synthetic marketplaces**

## 🎯 Overview

A research framework that treats LLMs as bounded policy approximators in a controlled marketplace environment. Study how cognitive architectures shape economic outcomes through systematic experimentation with freelancer and client agents.

### Key Research Capabilities

- **🧠 Strategic Decision-Making**: LLM agents develop bidding strategies through natural language reasoning
- **📈 Market Dynamics**: Track competition, efficiency, and inequality patterns over time  
- **🎭 Behavioral Adaptation**: Agents learn and adapt through reflection mechanisms
- **📊 Controlled Experiments**: Systematic comparison of agent configurations (LLM vs Random vs Hybrid)
- **🔬 Reproducible Results**: Complete framework available as open-source software

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key or custom LLM endpoint

### Installation
```bash
git clone [REPOSITORY_URL]
cd simulated_marketplace
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration
```bash
# Option A: Environment variable
export OPENAI_API_KEY='sk-your-key-here'

# Option B: Configuration file
cp config/private_config.example.py config/private_config.py
# Edit with your API credentials
```

### Run Simulation
```bash
# Test your connection
python test_llm_connection.py

# Run basic simulation
python run_marketplace.py --freelancers 20 --clients 5 --rounds 50

# Analyze results
python analyze_marketplace.py --simulation-file results/your_simulation.json
```

## 🏗️ Framework Architecture

### Core Components

**Simulation Engine**
- `TrueGPTMarketplace`: Main simulation with LLM-powered agents
- `SimpleReputationManager`: Multi-tier reputation system (New → Established → Expert → Elite)
- `JobRankingCalculator`: Semantic skill matching and job scoring

**Agent Types**
- **LLM Agents**: Strategic reasoning through structured prompts
- **Random Agents**: Probabilistic baseline (50% bid chance)
- **Greedy Agents**: Simple rational heuristics (highest budget preference)

**Analysis Tools**
- `MarketplaceAnalysis`: Comprehensive metrics and visualizations
- Statistical validation with confidence intervals
- Temporal trend analysis and adaptation tracking

### Key Features

**✅ Controlled Experimentation**
- Fixed population size for systematic comparison
- Configurable parameters (bid limits, reflection rates, cooldown periods)
- Multiple baseline agents for rigorous benchmarking

**✅ Advanced Reputation System**
- Performance-based tier progression
- Historical reputation tracking across rounds
- Reputation-aware decision making in agent prompts

**✅ Market Mechanics**
- Fixed-budget bidding (no rate negotiation)
- Natural skill distribution across job categories
- Bid cooloff system enabling re-bidding after N rounds
- Dynamic budget adjustments by clients

**✅ Type Safety & Validation**
- Pydantic models for all LLM responses (8 distinct types)
- Comprehensive error handling and logging
- 174+ automated tests with baseline agent validation

## 📊 Research Applications

### Comparative Studies
```bash
# Compare agent reasoning capabilities
python run_marketplace.py --baseline-scenario random     # Noise baseline
python run_marketplace.py --baseline-scenario greedy     # Rational baseline  
python run_marketplace.py                                # Full LLM agents

# Study reflection mechanisms
python run_marketplace.py --reflection-probability 0.0   # No reflections
python run_marketplace.py --reflection-probability 0.1   # Low reflection rate
```

### Market Mechanism Studies
```bash
# Bid cooloff effects
python run_marketplace.py --bid-cooloff-rounds 0    # No re-bidding
python run_marketplace.py --bid-cooloff-rounds 5    # 5-round cooloff

# Job posting frequency
python run_marketplace.py --job-posting-cooldown-min 1 --job-posting-cooldown-max 3   # High frequency
python run_marketplace.py --job-posting-cooldown-min 5 --job-posting-cooldown-max 15  # Low frequency
```

### Performance Optimization
```bash
# Large-scale experiments
python run_marketplace.py --freelancers 200 --clients 30 --rounds 100 --quiet --max-workers 20

# Accelerated progression
python run_marketplace.py --max-active-jobs 5  # Higher freelancer capacity
```

## 📈 Key Metrics Tracked

| Category | Metrics |
|----------|---------|
| **Efficiency** | Fill rate, bid efficiency, market health score |
| **Competition** | Bids per job, participation rate, selectivity |
| **Inequality** | Work distribution (Gini coefficient), tier distribution |
| **Adaptation** | Reputation progression, reflection patterns, strategy changes |
| **Market Health** | Saturation risk, engagement rates, recovery patterns |

## 🔬 Framework Assumptions

The framework makes several key assumptions to enable controlled experimentation:

**Market Structure**
- Fixed agent population (no entry/exit dynamics)
- Discrete time rounds with synchronous actions
- Binary hiring decisions (one freelancer per job)

**Agent Behavior**
- LLMs as bounded policy approximators with natural language reasoning
- Perfect memory through reflection system
- Reputation-aware decision making

**Economic Model**
- Fixed budgets with transparent pricing
- No transaction costs or platform fees
- Automatic job completion (success = True) to avoid evaluation bias

**Technical Implementation**
- Category-first job generation (business domains assigned to clients)
- Natural skill distribution (organic clustering around popular fields)
- Pure ranking system (no arbitrary threshold cutoffs)

*See [FRAMEWORK_ASSUMPTIONS.md](FRAMEWORK_ASSUMPTIONS.md) for complete details*

## 📊 Results & Analysis

All results are saved to `results/` directory:
- **Raw Data**: Complete interaction logs in JSON format
- **Analysis Reports**: Statistical summaries with confidence intervals  
- **Visualizations**: Publication-ready figures (market trends, agent learning, comparative analysis)
- **Reputation Tracking**: Historical progression data for longitudinal studies

## 🧪 Testing & Quality Assurance

Comprehensive testing ensures framework reliability:
```bash
# Run full test suite (174+ tests)
python -m pytest

# Test specific components
python -m pytest tests/test_baseline_agents.py      # Baseline agent validation
python -m pytest tests/test_marketplace_integration.py  # Integration tests
```

## 🔮 Research Directions

**Immediate Extensions**
- Cross-model validation (different LLM architectures)
- Human-AI hybrid markets
- Alternative auction mechanisms

**Long-term Opportunities**
- Real-world platform validation
- Policy implications research
- Large-scale economic ecosystem studies

## 🤝 Contributing

This research demonstrates AI-driven scientific discovery. Contributions welcome for:
- Extended research methodologies
- Technical improvements and optimizations  
- Validation studies and cross-platform testing
- Documentation and tutorial improvements

## 📜 Citation

```bibtex
TODO
```

---

*This research was conducted almost entirely by AI agents as part of the Agents4Science 2025 conference exploring AI-generated scientific research. The work represents a novel approach to scientific inquiry where artificial intelligence serves as both researcher and subject.*