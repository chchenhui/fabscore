# Reproducibility Statement

## Original NSRAG System
- **Determinism:** we set numpy/sklearn seeds; demo uses CPU-safe ops only
- **Data:** demo files are included in `data/demo/`. Full datasets are cited; scripts accept local paths
- **Intermediate Outputs:** the pipeline writes JSONL/CSV per stage in `outputs/`
- **Models:** the demo uses BM25 + template generator; production swaps in dense retrievers and HF LLMs via flags

## HDO System Implementation
- **Determinism:** HDO system uses controlled randomization for collusion resistance with configurable seeds. Random components include:
  - Routing entropy generation (cryptographically seeded)
  - Verifier selection tie-breaking (configurable randomization_factor=0.2)
  - Redundancy decisions (probability-based with configurable redundancy_prob)
  - Mock verifier responses in demo (for simulation purposes)
- **Data:** HDO demo scenarios are embedded in `scripts/hdo_demo.py` with 6 test cases covering medical, financial, code, educational, customer service, and research domains
- **Intermediate Outputs:** HDO system writes comprehensive outputs to `outputs/`:
  - `hdo_evaluation_report.txt`: Detailed performance analysis
  - `hdo_demo_results.json`: Episode data and metrics
  - Individual episode exports available via API
- **Models:** HDO uses mock verifiers for demonstration:
  - NLI verifiers (simulate cross-model ensemble)
  - Code verifiers (simulate static analysis tools)
  - Rule verifiers (policy checking)
  - Retrieval verifiers (knowledge base lookup)
  - Production deployment would integrate real NLI models, static analyzers, and knowledge bases
- **Configuration:** All HDO parameters are configurable via `HDOConfig`:
  - Uncertainty thresholds (tau_reject=0.2, tau_accept=0.8)
  - Delegation depth limits (max_delegation_depth=5)
  - Cost budgets and routing strategies
  - Collusion resistance settings
- **Evaluation:** Comprehensive metrics track alignment accuracy, efficiency gains, risk bounds, and paper claim verification
- **Reproducibility Verification:** Included `scripts/verify_reproducibility.py` demonstrates:
  - ✓ Deterministic mode produces identical results across runs
  - ✓ Configurable randomization for collusion resistance
  - ✓ Consistent episode hashing and result verification
  - ✓ Support for both reproducible research and production randomization