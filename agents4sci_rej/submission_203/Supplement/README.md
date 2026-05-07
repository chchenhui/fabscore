# <img src="assets/onesim.png" alt="YuLan Flower" width="35" style="vertical-align: middle;"> YuLan-OneSim (玉兰-万象)


<p align="center">
  <img src="assets/logo.gif" width="800"/>
</p>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2505.07581-b31b1b.svg)](https://arxiv.org/abs/2505.07581)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/Docs-online-brightgreen.svg)](https://ruc-gsai.github.io/YuLan-OneSim/)
[![Stars](https://img.shields.io/github/stars/RUC-GSAI/YuLan-OneSim)](https://github.com/RUC-GSAI/YuLan-OneSim/stargazers)


## YuLan-OneSim: A Next Generation Social Simulator with LLMs

</div>

## 📰 News
- **[2025-6-27]** 📢 Our documentation is now available at [YuLan-OneSim Docs](https://ruc-gsai.github.io/YuLan-OneSim/), and the Docker image can be accessed at [Docker Hub](https://hub.docker.com/repository/docker/ptss/yulan-onesim/general).
- **[2025-5-13]** 🎉 Initial release of YuLan-OneSim! Our paper is now available on [arXiv](https://arxiv.org/abs/2505.07581).

## 📋 Overview

YuLan-OneSim (玉兰-万象) is a groundbreaking social simulator that leverages Large Language Model (LLM) agents to model human social behaviors. Our framework bridges the gap between social science and artificial intelligence by offering a powerful, intuitive platform for research and exploration.

## ✨ Key Features

- 🔄 **Code-free scenario construction**: Design complex simulations through natural language conversations
  
- 📚 **Comprehensive default scenarios**: 50+ default scenarios across 8 major social science domains

- 🧠 **Evolvable simulation**: Models that automatically improve based on external feedback

- 🚀 **Large-scale simulation**: Distributed architecture supporting up to 100,000 agents

- 🔍 **AI social researcher**: Autonomous research from topic proposal to report generation


<p align="center">
  <img src="assets/overall.png" width="800"/>
</p>


## 🎥 Demo Video

<div align="center">
  <img src="assets/thumbnail.png" alt="YuLan-OneSim Demo Video" width="600"/>
  
  <p>
    <strong>📺 Watch Demo Video:</strong><br/>
  <a href="https://youtu.be/NUxZQleeEIc" target="_blank">中文版 </a> | 
  <a href="https://youtu.be/GSW2A76FIyw" target="_blank">English Version</a>
  </p>
</div>

## 🛠️ Installation

You can install YuLan-OneSim either using Docker (recommended for quick setup) or from source for development.

### 🐳 Docker Installation (Recommended)

We provide a Docker image for easy deployment and a `Makefile` to simplify common commands.

**Prerequisites:**
- [Docker](https://www.docker.com/products/docker-desktop) installed on your system.

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RUC-GSAI/YuLan-OneSim
    cd YuLan-OneSim
    ```

2.  **Configure your settings:**
    Before starting, make sure to set up your API keys and other settings in `config/config.json` and `config/model_config.json`. Refer to the [Configuration](#-configuration) section for details.

3.  **Run with Makefile (Easiest):**
    The `Makefile` provides simple commands to manage the container.
    ```bash
    make run
    ```
    Once running, access the application:
    - Web UI: [http://localhost:8000](http://localhost:8000)

    **Other useful `make` commands:**
    - `make build`: Build the Docker image.
    - `make run`: Start the container.
    - `make stop`: Stop and remove the container.
    - `make logs`: Follow container logs.
    - `make shell`: Enter the container's shell.
    - `make clean`: Stop the container and remove the image.

4.  **Run with Docker Commands:**
    If you prefer not to use `make`, you can use `docker` commands directly.
    ```bash
    # Pull the pre-built image from Docker Hub
    docker pull ptss/yulan-onesim:latest

    # Or build it from source
    # docker build -t ptss/yulan-onesim:latest .

    # Run the container
    docker run -d --name yulan-onesim -p 8000:80 -v ./config:/app/config ptss/yulan-onesim:latest
    ```

### 📦 Installation from Source

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/RUC-GSAI/YuLan-OneSim
cd YuLan-OneSim

# Install in editable mode
pip install -e .
# Install with tuning dependencies if needed
pip install -e .[tune]
```

## 🚀 Quick Start

### Command-Line Interface

```bash
# Run a simulation with default settings
yulan-onesim-cli --config config/config.json --model_config config/model_config.json --mode single --env labor_market_matching_process
```

### Web Server

```bash
# Start the backend API service (from the project root directory)
yulan-onesim-server

# In a new terminal, navigate to the frontend directory and start the frontend application
cd src/frontend
npm install # If you haven't installed dependencies
npm run dev # Or your specific command to start the frontend
```

Access the web interface (frontend) at `http://localhost:5173` and the API documentation (backend) at `http://localhost:8000/docs` .


<p align="center">
  <img src="assets/simulation.gif" width="800"/>
</p>

## ⚙️ Configuration

YuLan-OneSim uses JSON configuration files to control simulation behavior and model settings.

### Environment Configuration (.env files)

The frontend uses environment variables for configuration. Edit the following files in the `src/frontend` directory:

**.env.development**
```
# Development API endpoint (local development)
VITE_API_BASE_URL=http://localhost:8000
```

**.env.production**
```
# Production API endpoint (replace with your actual production server)
VITE_API_BASE_URL=https://your-production-server.com
```

These settings determine where the frontend will send API requests based on the environment. The `VITE_API_BASE_URL` variable is essential for connecting the frontend to the correct backend endpoint.

### Simulation Configuration (`config/config.json`)

The simulation configuration file controls general simulation parameters:

```json5
{
  "simulator": {
    "environment": {
      "name": "labor_market_matching_process",  // Environment scenario to simulate
      "mode": "round",                          // Simulation execution mode
      "max_steps": 3,                           // Maximum simulation steps
      "interval": 60.0,                         // Interval between simulation steps
      "export_training_data": false,            // Whether to export training data
      "export_event_data": false                // Whether to export event data
    }
  },
  "agent": {
    "planning": "COTPlanning",                  // Agent planning algorithm
    "memory": {
      "strategy": "ShortLongStrategy",          // Memory management strategy
      "storages": {
        "short_term_storage": {
          "class": "ListMemoryStorage",         // Short-term memory storage type
          "capacity": 100                       // Maximum memory capacity
        },
        "long_term_storage": {
          "class": "VectorMemoryStorage",       // Long-term memory storage type
          "capacity": 100,
          "model_config_name": "openai_embedding-bert"  // Embedding model for memory
        }
      },
      "metric_weights": {
        "recency": 0.7                          // Weight for recency in memory retrieval
      }
    }
  },
  "database": {
    "enabled": false,                           // Enable/disable database
    "host": "localhost",                        // Database host
    "port": 5432,                               // Database port
    "dbname": "onesim"                          // Database name
  },
  "distribution": {
    "enabled": false,                            // Enable/disable distributed mode
    "mode": "single",                           // Running mode: single/distributed
    "node_id": "1"                              // Node identifier for distributed mode
  },
  "monitor": {
    "enabled": true,                            // Enable/disable monitoring
    "update_interval": 30                       // Monitor update interval
  }
}
```

**Note:** The `model_config_name` values under `agent.memory` (e.g., for `long_term_storage` and the `relevance` metric) must match the `config_name` of an embedding model defined in `config/model_config.json`.

### Model Configuration (`config/model_config.json`)

Specifies the LLMs and embedding models used by the simulator:

```json5
{
  "chat": [
    {
      "provider": "openai",                     // LLM provider: OpenAI
      "config_name": "openai-gpt4o",            // Configuration identifier
      "model_name": "gpt-4o",                   // Model name
      "api_key": "sk-xxx",                      // API key
      "generate_args": {
        "temperature": 0                        // Temperature setting (0 = deterministic)
      }
    },
    {
      "provider": "vllm",                       // LLM provider: vLLM for local deployment
      "config_name": "vllm-qwen",               // Configuration identifier
      "model_name": "Qwen2.5-14B-Instruct",     // Model name or path
      "client_args": {
        "base_url": "http://localhost:9889/v1/" // Local API endpoint
      },
      "generate_args": {
        "temperature": 0                        // Temperature setting
      }
    }
  ],
  "embedding": [
    {
      "provider": "vllm",         // Embedding model type
      "config_name": "vllm_embedding-bert",   // Configuration identifier
      "model_name": "bge-base-en-v1.5",         // Embedding model name or path
      "client_args": {
        "base_url": "http://localhost:9890/v1/" // Local API endpoint
      }
    }
  ]
}
```

YuLan-OneSim supports both cloud-based APIs (like OpenAI) and locally deployed models through vLLM, giving users flexibility in their model choices. 

## 📁 Project Structure

```
├── config/        # Configuration files
├── src/           # Main source code
│   ├── onesim/    # Core simulation framework
│   ├── backend/   # FastAPI backend
│   ├── frontend/  # UI components
│   ├── llm_tuning/# Model fine-tuning
│   ├── envs/      # Simulation environments
│   └── researcher/# AI Social Researcher
└── scripts/       # Utility scripts
```


## 🧪 Experiments

1️⃣ Evaluation of the Automatic Scenario Generation Framework:

<p align="center">
  <img src="assets/ex-automatic.png" width="750"/>
</p>

We conducted a comprehensive evaluation of YuLan-OneSim's capabilities in automatic scenario generation. In terms of efficiency, the simulator achieves an average generation speed of 358 tokens per second. In terms of effectiveness, the quality scores for the generated agent behavior graphs and code both exceed 4 points (based on the scoring criteria described in the paper), demonstrating the system's strong potential in automatic scenario construction.

<p align="center">
  <img src="assets/ex-automatic-1.png" width="750"/>
</p>

Meanwhile, we observed that most of the errors generated by YuLan-OneSim are logical in nature. These include issues such as value access errors, mismatches between instructions and actions, incorrect value assignments, and omissions in type checking. Such errors can typically be resolved using standard debugging techniques and require minimal human intervention. However, their prevalence suggests that targeted improvements in logical verification could significantly enhance the reliability of the overall code generation process. Additionally, although syntax errors and robustness issues are also present in the generated outputs, they account for only a small portion of the total errors. While the current results are promising, our implementation still faces certain limitations when dealing with more complex logical dependencies and edge cases. To address these challenges, we plan to incorporate more advanced error detection and correction mechanisms in future work, aiming to further reduce the need for manual code adjustments.

2️⃣ Simulation Effectiveness Evaluation

To evaluate credibility, we conduct experiments from two perspectives: (1) Social Theory Validation — examining whether established classic social science theories can be validated within our simulation environment; (2) Real-World Data Alignment — assessing the degree of consistency between simulation results and real-world observational data.

<p align="center">
  <img src="assets/2.png" width="750"/>
</p>

In Experiment (1), we conduct simulations based on the Axelrod cultural dissemination model. As the simulation progresses, distinct cultural boundaries gradually emerge. Within each cultural region, neighboring agents exhibit high similarity (indicated by darker connection colors), while the boundaries between regions become increasingly clear. This visualization effectively reflects the core insight of Axelrod's theory: local interactions foster cultural homogeneity within regions, while cultural diversity is preserved on a global scale.


<p align="center">
  <img src="assets/2-1.png" width="750"/>
</p>

We also conducted a dynamic and quantitative analysis of the formation process underlying Axelrod's theory. As shown in the figure above, during the initial phase, the local convergence within communities gradually increases, while global polarization continues to decline—indicating that agents begin interacting and forming early-stage cultural clusters. Around the 15th iteration, an inflection point emerges: local convergence continues to rise, while global polarization stabilizes. This trend suggests that regions are becoming increasingly homogeneous internally, while the boundaries between different cultural groups remain clearly distinguishable.

These experimental results clearly illustrate the core of Axelrod's theory—the coexistence of local convergence and global polarization. YuLan-OneSim not only successfully reproduces this theoretical expectation but also provides a quantitative characterization of the process, offering deeper insights into the temporal dynamics of cultural dissemination.

<p align="center">
  <img src="assets/2-2.png" width="750"/>
</p>

In Experiment (2), we examine whether the data generated by the simulator aligns with real-world observations. Specifically, we conduct an experiment using housing market data from Brazil. Figure 8 presents a comparison between the simulated housing price distribution and the actual distribution in Brazil. The simulation successfully reproduces several key features observed in the real data—most notably, the multimodal distribution pattern with primary and secondary peaks that closely match the normalized values of the real distribution.

In the low-price range (0.05–0.25), the strong agreement between simulated and real data suggests that our model effectively captures the core dynamics of the most common housing market segment. Additionally, the simulation reflects the characteristic long-tail distribution often seen in housing markets. However, in the mid-range price interval (0.45–0.55), the simulation slightly underestimates the actual data. We attribute this discrepancy primarily to simplified modeling of the housing price formation process, which does not fully account for complex factors such as neighborhood livability, infrastructure quality, and historical valuations.

Despite this deviation, the overall results demonstrate that YuLan-OneSim can approximate real-world economic distribution patterns with high accuracy, validating its potential and practicality for applications in social science research.


3️⃣ Evaluation of the AI Social Researcher

<p align="center">
  <img src="assets/3.png" width="750"/>
</p>

<p align="center">
  <img src="assets/3-1.png" width="750"/>
</p>

In the evaluation of the AI Social Researcher, we focus primarily on the quality of the generated simulation scenario (ODD protocol) and the final analytical report. The results show that the AI Social Researcher performs well across all evaluation criteria for scenario design, achieving an average overall score of 4.13 out of 5. Notably, it excels in the "Feasibility" dimension, with an average score of 4.88, indicating its strong ability to translate abstract research questions into executable simulation plans. It also performs well in "Relevance" (average score of 4.25), ensuring that the generated scenarios are closely aligned with the user-defined research topics.

In terms of report generation, the AI Social Researcher demonstrates solid performance in "Structural Organization" (average score of 4.00) and "Content Completeness and Accuracy" (average score of 3.63), showing that it can effectively structure analysis results into logically coherent and well-supported technical reports. The top-performing reports include "Auction Market Dynamics" (Economics) and "Court Trial Simulation" (Law), each receiving an overall score of 4.0.

However, there remains room for improvement, particularly in the dimensions of "Insightfulness" (3.25) and "Practicality" (3.00). While the AI Social Researcher is competent in data analysis and reporting, it still struggles to extract deeper research insights and formulate actionable recommendations from simulation results. For instance, in the scenario "Labor Market Matching Process," the report received a low practicality score of 2.0, highlighting the current limitations of the model in certain domain-specific applications.




## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 Citation

If you use YuLan-OneSim in your research, please cite our paper:

```bibtex
@misc{wang2025yulanonesimgenerationsocialsimulator,
      title={YuLan-OneSim: Towards the Next Generation of Social Simulator with Large Language Models}, 
      author={Lei Wang and Heyang Gao and Xiaohe Bo and Xu Chen and Ji-Rong Wen},
      year={2025},
      eprint={2505.07581},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.07581}, 
}
```

## 🙏 Acknowledgments

We would like to express our sincere gratitude to the [AgentScope](https://github.com/modelscope/agentscope) team from Alibaba Tongyi Lab , whose outstanding work has made this project possible. Their contributions have provided a solid foundation and essential tools that greatly enhanced the development process.

Special thanks to [@pan-x-c](https://github.com/pan-x-c) (Xuchen Pan) and [@glendon84](https://github.com/glendon84) (Yaliang Li) for their invaluable guidance and support throughout the development of YuLan-OneSim.

## 📝 License

This project is licensed under the [Apache-2.0 License](LICENSE).