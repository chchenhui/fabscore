# LLM-Driven Catalyst Discovery Framework

This repository contains a comprehensive pipeline for discovering novel catalysts using Large Language Models (LLMs) combined with computational screening and validation.

## 📋 Overview

The framework implements the catalyst discovery approach described in the proposal, featuring:

- **Data Aggregation**: Collects catalyst data from Materials Project, NOMAD, OC20, and literature
- **RAG System**: Retrieval-Augmented Generation for grounding LLM outputs
- **Prompt Engineering**: Multiple generation strategies (constraint-based, analogy-based, combinatorial)
- **Screening**: Novelty checking and thermodynamic stability assessment
- **DFT Automation**: Automated setup and execution of DFT calculations
- **Feedback Loop**: Learning from validation results to improve future generations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd catalyst

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Set up API keys in environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export MP_API_KEY="your-materials-project-key"
```

2. Update configuration files:
- `config.json`: Data source configurations
- `pipeline_config.yaml`: Pipeline settings

### Running the Pipeline

Basic usage:
```bash
python scripts/catalyst_discovery_pipeline.py --reaction "CO2 reduction to CO" --elements Fe Co Ni Cu Mn
```

Iterative discovery:
```bash
python scripts/catalyst_discovery_pipeline.py --reaction "CO2 reduction" --iterative --max-elements 4
```

## 📁 Project Structure

```
catalyst/
├── scripts/
│   ├── data_aggregation.py         # Collect catalyst data
│   ├── embedding_indexing.py       # Build vector database
│   ├── rag_retrieval.py           # RAG system
│   ├── prompt_templates.py         # LLM prompt engineering
│   ├── novelty_screening.py       # Screen candidates
│   ├── dft_automation.py          # DFT calculation setup
│   ├── feedback_loop.py           # Learning system
│   └── catalyst_discovery_pipeline.py  # Main orchestration
├── data/                          # Data storage
│   ├── raw/                       # Aggregated data
│   ├── indexes/                   # Vector databases
│   └── knowledge_base/            # Feedback data
├── results/                       # Screening results
├── dft_calculations/             # DFT workspaces
├── pipeline_results/             # Pipeline outputs
├── config.json                   # Data source config
├── pipeline_config.yaml          # Pipeline config
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🔧 Individual Components

### 1. Data Aggregation
```bash
python scripts/data_aggregation.py
```
Collects catalyst data from multiple sources into a unified format.

### 2. Embedding and Indexing
```bash
python scripts/embedding_indexing.py --data-file data/raw/aggregated_catalyst_data_*.json
```
Creates searchable vector database from aggregated data.

### 3. RAG Retrieval
```bash
python scripts/rag_retrieval.py --query "high entropy alloy catalysts"
```
Retrieves relevant catalyst knowledge for LLM context.

### 4. Prompt Templates
```python
from prompt_templates import PromptTemplates, CatalystConstraints, GenerationStrategy

templates = PromptTemplates()
constraints = CatalystConstraints(
    allowed_elements=["Fe", "Co", "Ni"],
    max_elements=3
)

prompt = templates.build_generation_prompt(
    strategy=GenerationStrategy.CONSTRAINT_BASED,
    constraints=constraints,
    reaction="CO2 reduction",
    retrieved_context="...",
    num_candidates=5
)
```

### 5. Novelty Screening
```bash
python scripts/novelty_screening.py --candidates-file candidates.json --mp-api-key YOUR_KEY
```
Screens candidates for novelty and stability.

### 6. DFT Automation
```bash
python scripts/dft_automation.py --catalyst Cu --calculator vasp --adsorbates CO H
```
Sets up DFT calculations for validation.

### 7. Feedback Loop
```bash
python scripts/feedback_loop.py --action train
python scripts/feedback_loop.py --action report
```
Learns from validation results to improve generation.

## 📊 Output Examples

### Generated Candidates
```json
{
  "formula": "Fe0.25Co0.25Ni0.25Cu0.25",
  "structure": "fcc",
  "properties": {
    "expected_activity": "high",
    "expected_stability": "moderate"
  },
  "rationale": "High-entropy alloy with balanced d-band center",
  "similar_to": ["FeCo", "NiCu"]
}
```

### Screening Results
```json
{
  "formula": "Fe0.25Co0.25Ni0.25Cu0.25",
  "passed_screening": true,
  "checks": {
    "novelty": {
      "is_novel": true,
      "reason": "No similar materials found"
    },
    "stability": {
      "is_stable": true,
      "hull_distance": 0.05
    }
  }
}
```

## 🤝 Extending the Framework

### Adding New Data Sources
1. Implement data fetcher in `data_aggregation.py`
2. Update configuration in `config.json`
3. Ensure output format matches existing schema

### Adding Generation Strategies
1. Add new strategy to `GenerationStrategy` enum
2. Create template in `PromptTemplates`
3. Implement strategy-specific logic

### Custom DFT Workflows
1. Extend `DFTAutomation` class
2. Add calculator-specific input generation
3. Implement results parsing

## 📈 Performance Considerations

- **Data Collection**: Can take 10-30 minutes depending on sources
- **Embedding**: ~1-2 minutes per 10,000 materials
- **Generation**: ~30-60 seconds per strategy (with API calls)
- **Screening**: ~5-10 seconds per candidate
- **DFT**: Hours to days (depends on system size and calculator)

## 🐛 Troubleshooting

### Common Issues

1. **No Materials Project API key**
   - Sign up at https://materialsproject.org
   - Set `MP_API_KEY` environment variable

2. **OpenAI API errors**
   - Check API key is valid
   - Ensure sufficient credits
   - Pipeline works without API (uses mock data)

3. **DFT calculator not found**
   - Install required calculator (VASP/QE/GPAW)
   - Update paths in configuration

## 📚 References

Based on the catalyst discovery framework described in the proposal for LLM-driven materials discovery without fine-tuning.

## 📄 License

[Add appropriate license]