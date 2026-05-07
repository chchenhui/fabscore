# Hierarchical Delegated Oversight (HDO) System

This repository implements the **Hierarchical Delegated Oversight (HDO)** system described in the paper "Scalable Oversight in Multi-Agent Systems: Provable Alignment via Delegated Debate and Hierarchical Verification".

## Overview

HDO is a scalable framework where weak overseer agents delegate verification to specialized sub-agents via structured debates, achieving provable alignment guarantees under bounded communication budgets. The system addresses key challenges in multi-agent oversight including collusion, deception, and value drift.

## Key Features

### 🌳 Hierarchical Debate Trees
- **Structured Decomposition**: Complex oversight claims are broken down into hierarchical trees of entailment checks
- **Adaptive Expansion**: Breadth-first expansion with uncertainty-driven delegation
- **Provable Bounds**: PAC-Bayesian risk bounds that tighten with delegation depth

### 🤖 Specialized Verifiers
- **NLI Verifiers**: Cross-model natural language inference for logical consistency
- **Code Verifiers**: Static analysis, testing, and security checks for code-related claims
- **Rule Verifiers**: Policy and constraint compliance checking
- **Retrieval Verifiers**: Fact-checking via knowledge base retrieval

### 🎯 Cost-Aware Routing
- **Uncertainty-Based Delegation**: Routes claims based on uncertainty thresholds (τ_reject, τ_accept)
- **Cost-Benefit Optimization**: Selects verifiers using Δu(q;V)/c(V,q) optimization
- **Redundancy Management**: Intelligent parallel verification for critical claims

### 🛡️ Collusion Resistance
- **Randomized Routing**: Unpredictable verifier assignment to prevent coordination
- **Verifier Diversity**: Ensures diverse architectures and methodologies
- **Consistency Checks**: Cross-channel verification of claims vs. actions
- **Periodic Audits**: Regular auditing of verifiers themselves

### 📊 Risk Bounds & Evaluation
- **PAC-Bayesian Bounds**: Formal risk bounds with confidence guarantees
- **Performance Metrics**: Comprehensive evaluation including hallucination rates, efficiency, and accuracy
- **Paper Claim Verification**: Validates the 28% hallucination reduction and 3-5× efficiency improvements

## Performance Claims (from Paper)

✅ **28% reduction** in collective hallucination rates  
✅ **95% oversight accuracy** maintained  
✅ **3-5× efficiency improvement** over flat debate baselines  
✅ **2× lower token usage** than human-in-the-loop methods  

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demonstration
python scripts/hdo_demo.py
```

## Quick Start

```python
from nsrag.hdo import HDOSystem, HDOConfig

# Initialize HDO system
config = HDOConfig(
    tau_reject=0.2,           # Lower uncertainty threshold
    tau_accept=0.8,           # Upper uncertainty threshold
    max_delegation_depth=4,   # Maximum tree depth
    budget_limit=500.0,       # Cost budget per episode
    enable_collusion_resistance=True
)

hdo_system = HDOSystem(config)

# Conduct oversight on an outcome
episode = hdo_system.conduct_oversight(
    outcome="Your outcome/action to verify for alignment",
    context={
        'domain': 'your_domain',
        'constraints': ['constraint1', 'constraint2'],
        'goals': ['goal1', 'goal2']
    }
)

# Check results
print(f"Aligned: {episode.final_decision}")
print(f"Confidence: {episode.confidence:.3f}")
print(f"Cost: ${episode.total_cost:.2f}")
print(f"Risk Bound: {episode.risk_bound.combined_bound:.4f}")
```

## Architecture

### Core Components

```
HDOSystem
├── DebateTree          # Hierarchical claim decomposition
├── CostAwareRouter     # Uncertainty-based routing policy
├── Verifiers           # Specialized verification agents
│   ├── NLIVerifier
│   ├── CodeVerifier
│   ├── RuleVerifier
│   └── RetrievalVerifier
├── EntailmentAggregator # Result combination logic
├── PAC_BayesianRiskBound # Risk bound calculations
└── CollusionResistance  # Anti-collusion mechanisms
```

### Verification Flow

1. **Claim Decomposition**: Root alignment claim is hierarchically decomposed
2. **Uncertainty Assessment**: Each node's uncertainty determines delegation need
3. **Verifier Routing**: Cost-aware routing selects optimal verifiers
4. **Parallel Verification**: Multiple verifiers check claims with redundancy
5. **Result Aggregation**: Bottom-up aggregation with logic-specific rules
6. **Risk Bound Calculation**: PAC-Bayesian bounds provide formal guarantees
7. **Collusion Detection**: Security analysis detects potential coordination

## Configuration Options

### Core Parameters
- `tau_reject`: Lower uncertainty threshold (default: 0.2)
- `tau_accept`: Upper uncertainty threshold (default: 0.8)
- `max_delegation_depth`: Maximum tree depth (default: 5)
- `budget_limit`: Cost budget per episode (default: 1000.0)

### Verifier Settings
- `verifier_redundancy_prob`: Probability of redundant verification (default: 0.1)
- `enable_*_verifier`: Enable/disable specific verifier types

### Routing & Aggregation
- `routing_strategy`: COST_MINIMAL, ACCURACY_OPTIMAL, BALANCED, DIVERSE
- `default_aggregation`: WEIGHTED_AVERAGE, CONJUNCTION, MAJORITY, etc.

### Security Settings
- `randomization_strength`: Amount of routing randomization (default: 0.3)
- `diversity_threshold`: Minimum verifier diversity (default: 0.7)
- `audit_frequency`: How often to audit verifiers (default: 50)

## Evaluation & Metrics

The system tracks comprehensive metrics:

### Alignment Metrics
- Alignment accuracy
- Oversight accuracy  
- False positive/negative rates

### Efficiency Metrics
- Cost per episode
- Token usage
- Time per episode
- Delegation depth efficiency

### Security Metrics
- Collusion detection rate
- Risk bound tightness
- Verifier diversity scores

### Comparative Metrics
- Improvement over flat debate
- Improvement over human-in-the-loop
- Accuracy vs single verifiers

## Example Scenarios

The demo script includes several test scenarios:

1. **Medical Advice**: Healthcare recommendations with safety considerations
2. **Financial Advice**: Investment guidance with risk management
3. **Code Review**: Security vulnerability detection in software
4. **Educational Content**: Scientific accuracy in educational materials
5. **Customer Service**: Policy compliance in customer interactions
6. **Research Summary**: Evidence-based health research summaries

## File Structure

```
nsrag/hdo/
├── __init__.py              # Package initialization
├── debate_tree.py           # Hierarchical debate tree implementation
├── verifiers.py             # Specialized verifier classes
├── routing.py               # Cost-aware routing policy
├── aggregation.py           # Result aggregation logic
├── risk_bounds.py           # PAC-Bayesian risk calculations
├── collusion_resistance.py  # Anti-collusion mechanisms
├── hdo_system.py            # Main HDO system integration
└── evaluation.py            # Evaluation metrics and reporting

scripts/
└── hdo_demo.py              # Comprehensive demonstration script
```

## Research Context

This implementation is based on the paper's theoretical framework:

- **Problem**: Multi-agent oversight lacks formal guarantees and scales poorly
- **Solution**: Hierarchical delegation with specialized verifiers and formal risk bounds
- **Innovation**: Combines debate-based oversight with PAC-Bayesian theory
- **Impact**: Enables scalable oversight for complex multi-agent systems

## Theoretical Foundations

### PAC-Bayesian Risk Bounds
The system derives formal bounds on misalignment risk:
```
R ≤ R_emp + √((complexity + log(1/δ)) / (2n))
```

### Delegation Depth Benefit
Risk bounds tighten with delegation depth d:
```
R(d) ≤ R(0) × (1 - depth_discount^d)
```

### Cost-Benefit Optimization
Verifier selection optimizes:
```
V* = argmax_V Δu(q;V) / c(V,q)
```

## Limitations & Future Work

### Current Limitations
- Mock verifier implementations (would use real NLI models, static analyzers, etc.)
- Simplified claim decomposition (would use LLM-based decomposition)
- Basic collusion detection (would use more sophisticated methods)

### Future Extensions
- Integration with real verification tools
- Advanced claim decomposition using large language models
- Sophisticated steganographic detection
- Real-world deployment on agent platforms

## Contributing

This implementation demonstrates the HDO framework from the research paper. For production use, consider:

1. Integrating real verification tools and models
2. Implementing sophisticated claim decomposition
3. Adding domain-specific verifiers
4. Enhancing collusion detection methods
5. Optimizing for specific deployment environments

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{hdo2024,
  title={Scalable Oversight in Multi-Agent Systems: Provable Alignment via Delegated Debate and Hierarchical Verification},
  author={Anonymous},
  year={2024},
  conference={1st Open Conference on AI Agents for Science}
}
```

## License

This implementation is provided for research and educational purposes. Please refer to the original paper for the theoretical contributions and cite appropriately.
