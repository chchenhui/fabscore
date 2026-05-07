# Reproducibility Statement

This research utilized the YuLan-OneSim framework to investigate the joint effects of individual openness and information flow on cultural homogeneity through automated AI-driven simulation workflows. The complete research pipeline involved both human oversight and AI automation across five distinct phases.

## Research Question and Configuration

**Core Research Question**: "What is the joint impact of individuals' degree of openness and the degree of information flow on the cultural homogeneity that emerges in a society?"

**Project Configuration**:
- **Project Name**: `social_dynamics_combine`
- **Simulation Environment**: `src/envs/social_dynamics`
- **Research Paradigm**: Attribution Analysis Research
- **Framework**: YuLan-OneSim(Axelrod's cultural dissemination model)


## Step-by-Step Reproducibility Instructions

### Step 1: Environment Setup

**Prerequisites**:
```bash
# Clone the repository
cd YuLan-OneSim

# Install dependencies
pip install -e .

```

**Model Configuration Setup**:
```json
# Edit config/model_config.json
{
    "chat": [
        {
            "provider": "openai",
            "config_name": "gpt-4o",
            "model_name": "gpt-4o",
            "api_key": "xxxx",
            "client_args": {
                "base_url": "https://xxxx"
            },
            "generate_args": {
                "temperature": 0
            }
        }
    ],
    "embedding": [
        {
            "provider": "vllm",
            "config_name": "vllm-embedding-bert",
            "model_name": "/data/pretrain_dir/bge-base-en-v1.5",
            "api_key": "sk-xxx",
            "client_args": {
                "max_retries": 3,
                "base_url": "http://xxxx"
            }
        }
    ]
}
```


### Step 2: Phase 1 - Environment Design

**Command**:
```bash
python src/researcher.py --project_name "social_dynamics_combine" --phase design \
    --scenario "The higher the cultural similarity between individuals, the greater the probability they will interact and influence each other, leading to the convergence of cultural traits in local areas. However, this process may eventually lead to global cultural polarization (i.e., the formation of multiple stable cultural regions that are internally homogeneous but completely heterogeneous externally), rather than global homogenization. Individuals adjust their cultural traits by interacting with their neighbors. After multiple iterations, the society is observed to split into several cultural regions. Within each region, individual cultures are highly homogeneous, while there are significant differences between different regions. The focus is on observing the cultural homogeneity." \
    --question "What is the joint impact of individuals' degree of openness and the degree of information flow on the cultural homogeneity that emerges in a society?" \
    --paradigm "attribution_analysis"
```

**What Happens**:
1. **AI Automated Process**:
   - Creates `projects/social_dynamics_combine/` directory structure
   - Generates `src/envs/social_dynamics/scene_info.json` with ODD protocol
   - Creates experimental design in `experiment_design/experiment_config.json`
   - Designs 3×3 factorial experiment (9 groups total)

2. **Human Role**: Provide research scenario and question as input parameters
3. **AI Role**: Automatically convert natural language descriptions into formal simulation specifications


### Step 3: Phase 2 - Scenario Creation

**Command**:
```bash
python src/researcher.py --project_name "social_dynamics_combine" --phase scenario
```

**What Happens**:
1. **AI Automated Process**:
   - Reads environment design from Phase 1
   - Generates complete agent behavior code
   - Creates cultural dissemination algorithms
   - Generates agent profile schemas and relationship matrices

2. **Human Role**: Monitor execution, verify no errors occurred
3. **AI Role**: Code generation, algorithm implementation, metrics design

**Key Outputs**:
- Complete simulation environment code in `src/envs/social_dynamics/`
- Agent behavior implementations
- Metrics calculation functions
- Network topology definitions

### Step 4: Phase 3 - Experiment Execution

**Command**:
```bash
python src/researcher.py --project_name "social_dynamics_combine" --phase execute
```

**What Happens**:
1. **AI Automated Process**:
   - Executes 9 experimental groups simultaneously
   - Applies parameter variations (openness: very_low/medium/very_high × information_flow: first/third/fifth-order)
   - Runs 100-agent simulations for each group
   - Collects time-series data for metrics
   - Saves detailed execution logs and raw outputs

2. **Human Role**: Monitor system resources, verify completion status
3. **AI Role**: Complete experimental execution, data collection, error handling


### Step 5: Phase 4 - Data Analysis

**Command**:
```bash
python src/researcher.py --project_name "social_dynamics_combine" --phase analysis
```

**What Happens**:
1. **AI Automated Process**:
   - Locates simulation results across all 9 experimental groups
   - Performs statistical analysis on cultural homogeneity metrics
   - Generates comparative analysis between treatment conditions
   - Creates visualization plots and charts
   - Computes effect sizes and significance tests

2. **Analysis Methods**:
   - Cultural Homogeneity Index calculations
   - Statistical comparisons between openness levels
   - Information flow impact analysis
   - Interaction effect detection

3. **Human Role**: Review analysis results, provide feedback on interpretations
4. **AI Role**: Automated statistical analysis, pattern recognition, visualization generation


### Step 6: Phase 5 - Report Generation

**Command**:
```bash
python src/researcher.py --project_name "social_dynamics_combine" --phase report
```

**What Happens**:
1. **AI Automated Process**:
   - Synthesizes results from all previous phases
   - Generates comprehensive research report
   - Creates LaTeX-formatted academic paper
   - Includes statistical analysis and visualizations
   - Produces multiple output formats

2. **Content Generation**:
   - Introduction and literature review
   - Methodology and experimental design
   - Results and statistical analysis
   - Discussion and conclusions
   - Automated figure generation and referencing

3. **Human Role**: Review generated content, provide feedback, adjust formatting
4. **AI Role**: Complete content generation, figure creation, document formatting

**Key Outputs**:
- Complete research report in multiple formats
- Academic paper with statistical analysis
- Figure generation and visualization
- Bibliography and citations

## Verification and Validation


## Expected Results Structure

After successful completion, the project structure contains:

```
projects/social_dynamics_combine/
├── README.md                    # Human-AI collaborative research design
├── workflow_state.json          # AI-tracked execution state
├── experiment_design/
│   └── experiment_config.json   # AI-generated experimental parameters
├── base_scenario/               # AI-generated environment specifications
├── groups/                      # AI-executed experimental data
│   ├── control_group_very_low_openness_first_order/
│   ├── treatment_medium_openness_fifth_order/
│   ├── treatment_medium_openness_first_order/
│   ├── treatment_medium_openness_third_order/
│   ├── treatment_very_high_openness_fifth_order/
│   ├── treatment_very_high_openness_first_order/
│   ├── treatment_very_high_openness_third_order/
│   ├── treatment_very_low_openness_fifth_order/
│   └── treatment_very_low_openness_third_order/
├── analysis/                    # AI-generated statistical analysis
└── reports/                     # AI-generated research reports
```


This reproducibility statement provides a complete roadmap for replicating the AI-driven research workflow, with clear delineation of human and AI contributions at each phase.