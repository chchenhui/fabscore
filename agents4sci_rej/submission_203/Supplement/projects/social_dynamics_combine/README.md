# social_dynamics_combine

## Project Overview

- **Research Paradigm**: Attribution Analysis Research
- **Paradigm Description**: Analyze relative influence weights of different factors
- **Scenario Description**: The higher the cultural similarity between individuals, the greater the probability they will interact and influence each other, leading to the convergence of cultural traits in local areas. However, this process may eventually lead to global cultural polarization (i.e., the formation of multiple stable cultural regions that are internally homogeneous but completely heterogeneous externally), rather than global homogenization. Individuals adjust their cultural traits by interacting with their neighbors. After multiple iterations, the society is observed to split into several cultural regions. Within each region, individual cultures are highly homogeneous, while there are significant differences between different regions. The focus is on observing the cultural homogeneity.
- **Research Question**: What is the joint impact of individuals' degree of openness and the degree of information flow on the cultural homogeneity that emerges in a society?
Here, “individuals’ degree of openness” refers to a behavioral parameter — in conjunction with cultural similarity — that determines whether an individual adopts a neighbor’s cultural trait. Meanwhile, “degree of information flow” refers to the spatial range of interaction, defined by the order of neighbors (e.g., first-order = immediate N/S/E/W; 2nd/third-order = extended neighbors) with whom an agent can communicate. While the original model restricts both adoption propensity (via fixed openness) and interaction range (only first-order neighbors), our extended framework allows independent and simultaneous variation of both parameters, enabling a more nuanced exploration of how psychological receptivity and structural connectivity jointly shape cultural homogeneity.
- **Created**: 2025-09-09 17:46:48

## Research Design Plan

### Phase 1: Environment Design (env_design)
- **Goal**: Build simulation environment based on scenario description and research question
- **Paradigm Adaptation**: Multi-factor modeling, design sensitivity analysis experiments
- **Expected Outputs**: 
  - Scene configuration files
  - Agent behavior rules
  - Experimental design specifications

### Phase 2: Scenario Creation (scenario_creation)  
- **Goal**: Generate complete executable simulation scenario
- **Approach**: Consistent with existing workflow
- **Expected Outputs**: Complete simulation environment code

### Phase 3: Experiment Execution (experiment_execution)
- **Goal**: Execute simulation experiments and collect data
- **Experimental Design**: Factor analysis experiments to measure relative influences
- **Expected Outputs**: Experimental data and preliminary results

### Phase 4: Data Analysis (analysis)
- **Goal**: Deep analysis of experimental results
- **Analysis Focus**: Factor importance ranking and interaction effect analysis
- **Expected Outputs**: Statistical analysis and pattern recognition results

### Phase 5: Report Generation (report_generation)
- **Goal**: Generate research reports
- **Report Style**: Analytical research report highlighting factor influence weights
- **Expected Outputs**: Complete research report documents

## Project Structure

```
social_dynamics_combine/
├── README.md                   # This file with design plan
├── experiment_design/          # Experimental design files
├── base_scenario/             # Base scenario configuration
├── groups/                    # Experimental group data
├── analysis/                  # Analysis results
└── reports/                   # Generated reports
```

## Technical Configuration

### Model Configuration
- **Model Name**: openai-gpt4.1
- **Config Path**: config/model_config.json

## Usage

### Full Workflow Execution
```bash
python src/researcher.py \
    --project_name "social_dynamics_combine" \
    --scenario "The higher the cultural similarity between individuals, the greater the probability they will interact and influence each other, leading to the convergence of cultural traits in local areas. However, this process may eventually lead to global cultural polarization (i.e., the formation of multiple stable cultural regions that are internally homogeneous but completely heterogeneous externally), rather than global homogenization. Individuals adjust their cultural traits by interacting with their neighbors. After multiple iterations, the society is observed to split into several cultural regions. Within each region, individual cultures are highly homogeneous, while there are significant differences between different regions. The focus is on observing the cultural homogeneity." \
    --question "What is the joint impact of individuals' degree of openness and the degree of information flow on the cultural homogeneity that emerges in a society?
Here, “individuals’ degree of openness” refers to a behavioral parameter — in conjunction with cultural similarity — that determines whether an individual adopts a neighbor’s cultural trait. Meanwhile, “degree of information flow” refers to the spatial range of interaction, defined by the order of neighbors (e.g., first-order = immediate N/S/E/W; 2nd/third-order = extended neighbors) with whom an agent can communicate. While the original model restricts both adoption propensity (via fixed openness) and interaction range (only first-order neighbors), our extended framework allows independent and simultaneous variation of both parameters, enabling a more nuanced exploration of how psychological receptivity and structural connectivity jointly shape cultural homogeneity." \
    --paradigm "attribution_analysis"
```

### Phase-by-Phase Execution
```bash
# Environment design only
python src/researcher.py --project_name "social_dynamics_combine" --phase design

# Complete scenario creation
python src/researcher.py --project_name "social_dynamics_combine" --phase scenario

# Experiment and analysis
python src/researcher.py --project_name "social_dynamics_combine" --phase execute
python src/researcher.py --project_name "social_dynamics_combine" --phase analysis

# Report generation
python src/researcher.py --project_name "social_dynamics_combine" --phase report
```

## Expected Outcomes


### Attribution Analysis Expected Outcomes
- 📊 Factor importance rankings
- 📊 Relative influence weights
- 📊 Interaction effect analysis
- 📊 Optimal configuration recommendations


---

*Generated by OneSim Multi-Paradigm Research Workflow*
