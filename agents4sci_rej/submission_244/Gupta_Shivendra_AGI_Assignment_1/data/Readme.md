# Data directory

**Why is this folder (mostly) empty?**
- The **synthetic dataset is generated in code** by `ResearchDataset` inside `submission/code/implementation.py`.
  No downloads or files are required to run the NumPy simulation.
- The **SST-2** experiment (optional) is fetched at runtime via HuggingFace `datasets` in
  `submission/code/experiment_sst2.ipynb`. We do **not** redistribute SST-2.