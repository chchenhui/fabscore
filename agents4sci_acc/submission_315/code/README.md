# Magellan: Guided MCTS for Latent Space Exploration and Novelty Generation

This repository is the official implementation of the paper **"Magellan: Guided MCTS for Latent Space Exploration and Novelty Generation"**.

Magellan is a novel framework that reframes creative generation as a principled, guided exploration of a Large Language Model's (LLM) latent conceptual space. At its core, Magellan employs Monte Carlo Tree Search (MCTS) governed by a hierarchical guidance system to generate novel and plausible scientific ideas, significantly outperforming strong baselines like ReAct and Tree of Thoughts (ToT).

## Framework Overview

Magellan steers an LLM away from its training data's "gravity wells" towards innovative ideas through a two-level guidance system:

1.  **Strategic Compass**: A "semantic compass" vector, formulated via orthogonal projection, sets a long-range direction for the search, steering it towards relevant novelty.

2.  **Tactical Navigation**: A landscape-aware value function provides step-by-step guidance, replacing flawed self-evaluation with an explicit reward structure that balances coherence, novelty, and progress.

This principled, guided-search approach proves more effective for creative discovery than the unconstrained agency found in other methods.

## Setup

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (for FAISS-GPU and model inference)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/magellan.git
    cd magellan
    ```

2.  **Install dependencies:**
    The requirements are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The default installation uses `faiss-gpu`. If you do not have a compatible GPU setup, please modify `requirements.txt` to use `faiss-cpu` instead.*

3.  **Download Model Weights:**
    This project requires access to Qwen model weights (e.g., `Qwen1.5-1.8B`). Please download them from a legitimate source (e.g., Hugging Face). All scripts that require a model path have a `--modelpath` argument where you can specify the location of the weights.

## Usage

The workflow is divided into three main stages: Database Preparation, Running Experiments, and Analysis.

### 1. Database Preparation

First, we need to construct the knowledge corpus from scientific papers.

**a) Collect Paper Data:**
Scripts are provided to scrape abstracts from sources like CVPR, ICML, and Nature Medicine.

```bash
# Example for ICML
python ./database/code/scrape_openreview_papers.py --year=2025 --conference='ICML' --output_dir='./database/data_use'
```

```bash
# Example for CVPR
python ./database/code/scrape_cvpr.py --year=2025 --conference='CVPR' --output_dir='./database/data_use'
```
```bash
# Example for Nature Medicine
python ./database/code/scrape_nature_medicine.py --start_year=2022 --end_year=2025 --output_dir='./database/data_use'
```


**b) Build Vector Database:**
Encode the scraped papers into a FAISS vector database.

```bash
python ./database/build_database.py --modelpath /path/to/your/qwen/weights --inputpath './database/raw_data' --outdir './database/vector_db'
```

This will generate `faiss_index.bin`, `paper_metadata.json`, and `vectors.npy` in the output directory.

**c) Construct Test Set:**
Generate the set of test themes used for evaluation.

```bash
python ./database/construct_test_dataset.py --modelpath /path/to/your/qwen/weights --dbpath './database/vector_db' --outdir './data/test_set'
```

### 2. Running Experiments

All experiment scripts are located in the `./experiments/` directory. You can run experiments for Magellan and various baselines.

```bash
# Example for running the Magellan experiment
python ./experiments/run_experiments_magellan.py \
    --modelpath /path/to/your/qwen/weights \
    --dbpath './database/vector_db' \
    --testfile './data/test_set/test_themes.json' \
    --outfile './results/results_magellan.json'
```

### 3. Analysis

We provide scripts in the `./analysis/` folder to help evaluate the results.

- `llm_judge.py`: An example script for using an "LLM as a Judge" to score the outputs. You may need to adapt the prompts based on the models you are comparing.

- `analyze_results.py`: An example script to parse the judge's scores and compute final metrics.

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{your_name_2025_magellan,
  title={Magellan: Guided MCTS for Latent Space Exploration and Novelty Generation},
  author={Your Name, et al.},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

## License

This project is licensed under the [CC BY-NC-SA 4.0 License](LICENSE).