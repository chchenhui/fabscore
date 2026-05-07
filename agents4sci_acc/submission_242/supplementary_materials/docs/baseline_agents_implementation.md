# Baseline Agents Implementation

## Overview

This implementation adds comprehensive baseline agent types to the TrueGPTMarketplace simulation, providing simple, deterministic alternatives to LLM-powered agents for benchmarking and comparison purposes.

## Baseline Agent Types

### 1. Random Agents (`random`)
**Pure noise baseline - minimal intelligence**
- **Freelancer Behavior**: Configurable probability to bid on any shown job (50% default)
- **Bid Pricing**: Random amount between 50-150% of job budget (uniform distribution)
- **Client Behavior**: 50% chance to hire, randomly selects among received bids when hiring
- **Use Case**: Noise floor baseline to verify LLM agents outperform random behavior

### 2. Greedy Agents (`greedy`)
**Simple rational heuristic**
- **Freelancer Behavior**: Always bids on the highest budget jobs first
- **Bid Pricing**: Undercuts by percentage (90% of budget by default)
- **Client Behavior**: Always selects the lowest-priced bid within budget
- **Use Case**: Simple rational baseline that exploits budget visibility

### 3. No-Reputation Agents (`no_reputation`)
**Greedy strategy ignoring reputation**
- **Behavior**: Same as greedy agents
- **Key Difference**: Completely ignores reputation system
- **Use Case**: Isolates the value of reputation scaffolding in the system

## Command Line Usage

### Pre-configured Scenarios

```bash
# Pure noise baseline
python run_marketplace.py --baseline-scenario random

# Simple rational baseline  
python run_marketplace.py --baseline-scenario greedy

# Reputation-agnostic baseline
python run_marketplace.py --baseline-scenario no_reputation

# LLM without reputation (still under development)
python run_marketplace.py --baseline-scenario llm_no_reputation

# LLM without reflections
python run_marketplace.py --baseline-scenario llm_no_reflection
```

### Custom Configuration

```bash
# Mixed agent types
python run_marketplace.py \
  --freelancer-agent-type greedy \
  --client-agent-type random \
  --baseline-greedy-undercut 0.85

# Custom undercut percentage for greedy agents
python run_marketplace.py \
  --freelancer-agent-type greedy \
  --baseline-greedy-undercut 0.75  # 75% of budget

# Custom bid probability for random agents
python run_marketplace.py \
  --freelancer-agent-type random \
  --random-freelancer-bid-probability 0.2  # 20% chance to bid
```

## Bid Pricing Behavior

### Random Agents
- **Fixed Strategy**: Random bid between 50-150% of job budget (uniform distribution)
- Provides price variation and competition

### Greedy Agents  
- **Fixed Strategy**: Always undercut by percentage (90% of budget by default)
- Competitive pricing strategy designed to win jobs
- Customizable via `--baseline-greedy-undercut` parameter

## Key Features

### Deterministic Behavior
- Greedy agents produce identical results given same inputs
- Random agents use seeded randomness for reproducible experiments
- No reflection or learning - pure policy execution

### Job Selection Logic
- **Greedy/No-reputation agents**: Always see highest budget jobs first
- **Random/LLM agents**: Use original selection method (random or relevance-based)

### Performance Optimized
- No API calls for baseline agents
- Reflections automatically disabled for non-LLM scenarios
- Significantly faster execution than LLM agents

### Realistic Market Behavior
- **Random agents** now implement realistic fill rates:
  - Freelancers: Configurable probability to bid (50% default - markets aren't fully saturated)
  - Clients: 50% chance to hire (not every job gets filled, even with bids)
  - Results in configurable overall fill rates (default: ~25% for pure random markets = 0.5 × 0.5)
- **Greedy agents** maintain high fill rates through rational decision-making:
  - Always bid on highest-value opportunities
  - Always hire the most competitive bid
- Provides range of fill rates for comparison: Random (5%-50%) → Mixed → Greedy (~100%)

## Testing

Comprehensive test suite in `tests/test_baseline_agents.py`:

```bash
# Run all baseline agent tests
python -m pytest tests/test_baseline_agents.py -v

# Test specific functionality
python -m pytest tests/test_baseline_agents.py::TestFreelancerBiddingBaselines -v
```

## Integration with Existing System

### Metrics Compatibility
- All existing marketplace metrics work with baseline agents
- Same data structures and analysis pipelines
- Compatible with comparative analysis tools

### Caching
- Baseline agents work with or without caching enabled
- Persona generation still uses LLM/cache for consistency
- Only decision-making logic is replaced

### Parallel Processing
- Maintains full parallelization support
- Baseline decisions processed in parallel for performance
- Same worker pool configuration

## Comparative Analysis Workflow

### Suggested Comparison Order

1. **Random baseline** (noise floor)
2. **Greedy baseline** (simple rationality) 
3. **LLM without reputation/reflection** (stripped-down LLM)
4. **Full LLM agents** (complete system)

### Key Metrics to Compare

- Fill rate (jobs successfully filled)
- Bids per job (competition intensity)
- Success rate for freelancers
- Market health score
- Work distribution (Gini coefficient)

### Sample Commands for Comparative Studies

```bash
# Generate datasets for comparison (same seed for reproducibility)
python run_marketplace.py --baseline-scenario random --rounds 50 --freelancers 10 --quiet
python run_marketplace.py --baseline-scenario greedy --rounds 50 --freelancers 10 --quiet  
python run_marketplace.py --freelancers 10 --rounds 50 --quiet  # Full LLM
```

## Implementation Details

### Clean Integration
- New agent types integrate through existing decision dispatch methods
- No changes to core marketplace logic or entities
- Backward compatible with existing configurations

### Error Handling
- Robust validation of agent type parameters
- Clear error messages for invalid configurations
- Graceful fallbacks for edge cases

### Code Organization
- Baseline methods clearly separated with `_baseline_` prefixes
- Comprehensive documentation and type hints
- Follows existing code patterns and conventions

This implementation provides a solid foundation for benchmarking LLM-driven marketplace behavior against well-understood baselines, enabling rigorous evaluation of the value added by LLM scaffolding and complex agent behaviors.
