#!/usr/bin/env python3
"""
Comprehensive Dataset Download Script for LLM Inbreeding Deterioration Analysis

This script downloads multiple public research datasets relevant to analyzing
LLM quality degradation over iterative training cycles.
"""

import os
import sys
import json
import requests
import zipfile
import tarfile
import gdown
from pathlib import Path
from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def create_directories():
    """Create necessary directory structure"""
    dirs = [
        'data/raw',
        'data/processed', 
        'data/evaluation',
        'data/reasoning',
        'data/coding',
        'data/knowledge'
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Created directory structure")

def download_with_progress(url, filepath, desc="Downloading"):
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as file, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        print(f"✗ Failed to download {url}: {e}")
        return False

def download_huggingface_dataset(dataset_name, subset=None, split=None, save_path=None):
    """Download dataset from Hugging Face"""
    try:
        print(f"📦 Downloading {dataset_name}...")
        if subset:
            dataset = load_dataset(dataset_name, subset)
        else:
            dataset = load_dataset(dataset_name)
        
        if save_path:
            # Save as CSV for easy inspection
            if split and split in dataset:
                df = pd.DataFrame(dataset[split])
                df.to_csv(save_path, index=False)
            elif 'train' in dataset:
                df = pd.DataFrame(dataset['train'])
                df.to_csv(save_path, index=False)
            else:
                # Save all splits
                for split_name, split_data in dataset.items():
                    df = pd.DataFrame(split_data)
                    base_path = save_path.replace('.csv', f'_{split_name}.csv')
                    df.to_csv(base_path, index=False)
        
        print(f"✓ Downloaded {dataset_name}")
        return dataset
    except Exception as e:
        print(f"✗ Failed to download {dataset_name}: {e}")
        return None

def main():
    print("🚀 Starting comprehensive dataset download for LLM inbreeding analysis...")
    print("=" * 80)
    
    create_directories()
    
    dataset_info = []
    
    # 1. EVALUATION BENCHMARKS
    print("\n📊 Downloading Evaluation Benchmarks...")
    
    # MMLU - Massive Multitask Language Understanding
    mmlu = download_huggingface_dataset(
        "cais/mmlu", 
        subset="all",
        save_path="data/evaluation/mmlu.csv"
    )
    if mmlu:
        dataset_info.append({
            "name": "MMLU",
            "path": "data/evaluation/mmlu.csv", 
            "size": "~25MB",
            "description": "57 academic subjects for testing knowledge",
            "samples": "~15,000 questions",
            "relevance": "Core benchmark for measuring knowledge retention across training iterations"
        })
    
    # HellaSwag - Commonsense reasoning
    hellaswag = download_huggingface_dataset(
        "Rowan/hellaswag",
        save_path="data/evaluation/hellaswag.csv"
    )
    if hellaswag:
        dataset_info.append({
            "name": "HellaSwag", 
            "path": "data/evaluation/hellaswag.csv",
            "size": "~35MB",
            "description": "Commonsense reasoning completion tasks",
            "samples": "~70,000 examples",
            "relevance": "Tests if model maintains reasoning ability through iterations"
        })
    
    # ARC - AI2 Reasoning Challenge
    arc = download_huggingface_dataset(
        "allenai/ai2_arc",
        subset="ARC-Challenge", 
        save_path="data/evaluation/arc.csv"
    )
    if arc:
        dataset_info.append({
            "name": "ARC",
            "path": "data/evaluation/arc.csv",
            "size": "~5MB", 
            "description": "Science exam questions requiring reasoning",
            "samples": "~7,787 questions",
            "relevance": "Measures reasoning degradation in scientific domains"
        })
    
    # 2. MATHEMATICAL REASONING
    print("\n🧮 Downloading Mathematical Reasoning Datasets...")
    
    # GSM8K - Grade School Math
    gsm8k = download_huggingface_dataset(
        "openai/gsm8k",
        subset="main",
        save_path="data/reasoning/gsm8k.csv"
    )
    if gsm8k:
        dataset_info.append({
            "name": "GSM8K",
            "path": "data/reasoning/gsm8k.csv", 
            "size": "~8MB",
            "description": "Grade school math word problems",
            "samples": "~8,500 problems",
            "relevance": "Critical for measuring mathematical reasoning preservation"
        })
    
    # MATH Dataset
    try:
        math_dataset = download_huggingface_dataset(
            "lighteval/MATH",
            save_path="data/reasoning/math.csv"
        )
        if math_dataset:
            dataset_info.append({
                "name": "MATH",
                "path": "data/reasoning/math.csv",
                "size": "~15MB", 
                "description": "Competition mathematics problems",
                "samples": "~12,500 problems",
                "relevance": "Advanced mathematical reasoning benchmark"
            })
    except:
        print("⚠️  MATH dataset unavailable, skipping...")
    
    # 3. CODE GENERATION 
    print("\n💻 Downloading Code Generation Datasets...")
    
    # HumanEval - Code generation benchmark  
    humaneval = download_huggingface_dataset(
        "openai/openai_humaneval",
        save_path="data/coding/humaneval.csv"
    )
    if humaneval:
        dataset_info.append({
            "name": "HumanEval",
            "path": "data/coding/humaneval.csv",
            "size": "~1MB",
            "description": "Python programming problems", 
            "samples": "164 problems",
            "relevance": "Tests code generation quality through iterations"
        })
    
    # MBPP - Mostly Basic Python Problems
    try:
        mbpp = download_huggingface_dataset(
            "google-research-datasets/mbpp",
            subset="full",
            save_path="data/coding/mbpp.csv" 
        )
        if mbpp:
            dataset_info.append({
                "name": "MBPP",
                "path": "data/coding/mbpp.csv",
                "size": "~3MB",
                "description": "Basic Python programming problems",
                "samples": "~1,000 problems", 
                "relevance": "Complementary coding benchmark"
            })
    except:
        print("⚠️  MBPP dataset unavailable, skipping...")
    
    # 4. KNOWLEDGE BENCHMARKS
    print("\n📚 Downloading Knowledge Benchmarks...")
    
    # TruthfulQA - Truthfulness
    truthfulqa = download_huggingface_dataset(
        "truthfulqa/truthful_qa",
        subset="generation",
        save_path="data/knowledge/truthfulqa.csv"
    )
    if truthfulqa:
        dataset_info.append({
            "name": "TruthfulQA", 
            "path": "data/knowledge/truthfulqa.csv",
            "size": "~2MB",
            "description": "Questions testing truthfulness vs falsehoods",
            "samples": "~817 questions",
            "relevance": "Critical for measuring factual accuracy preservation"
        })
    
    # WinoGrande - Commonsense reasoning
    winogrande = download_huggingface_dataset(
        "allenai/winogrande", 
        subset="winogrande_xl",
        save_path="data/knowledge/winogrande.csv"
    )
    if winogrande:
        dataset_info.append({
            "name": "WinoGrande",
            "path": "data/knowledge/winogrande.csv",
            "size": "~10MB", 
            "description": "Commonsense reasoning with pronouns",
            "samples": "~44,000 examples",
            "relevance": "Tests commonsense reasoning stability"
        })
    
    # 5. ADDITIONAL DATASETS VIA DIRECT DOWNLOAD
    print("\n🌐 Downloading Additional Datasets...")
    
    # SuperGLUE components via direct download
    superglue_tasks = {
        "boolq": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip",
        "copa": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/COPA.zip", 
        "rte": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip"
    }
    
    for task, url in superglue_tasks.items():
        filepath = f"data/raw/{task}.zip"
        if download_with_progress(url, filepath, f"Downloading {task.upper()}"):
            dataset_info.append({
                "name": f"SuperGLUE-{task.upper()}",
                "path": filepath,
                "size": "~1-5MB",
                "description": f"SuperGLUE {task} task",
                "samples": "Variable",
                "relevance": "Language understanding benchmark"
            })
    
    # Create comprehensive dataset documentation
    print("\n📋 Creating dataset documentation...")
    
    # Save dataset inventory
    with open('data/dataset_inventory.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Create comprehensive README
    readme_content = f"""# LLM Inbreeding Deterioration Analysis - Datasets

This folder contains {len(dataset_info)} datasets for analyzing quality degradation in iteratively trained LLMs.

## Dataset Categories

### 🔍 Evaluation Benchmarks
Core benchmarks for measuring model capabilities:
- **MMLU**: Massive Multitask Language Understanding (57 academic subjects)
- **HellaSwag**: Commonsense reasoning completion tasks  
- **ARC**: AI2 Reasoning Challenge (science questions)

### 🧮 Mathematical Reasoning
Datasets testing mathematical and logical reasoning:
- **GSM8K**: Grade school math word problems
- **MATH**: Competition mathematics problems

### 💻 Code Generation  
Programming and code synthesis benchmarks:
- **HumanEval**: Python programming problems
- **MBPP**: Mostly Basic Python Problems

### 📚 Knowledge & Truthfulness
Testing factual knowledge and truthfulness:
- **TruthfulQA**: Questions testing truthfulness
- **WinoGrande**: Commonsense reasoning with pronouns

### 🎯 Language Understanding
Additional SuperGLUE tasks:
- **BoolQ**: Yes/no questions
- **COPA**: Choice of plausible alternatives
- **RTE**: Recognizing textual entailment

## Dataset Sizes
"""
    
    total_size_estimate = 0
    for dataset in dataset_info:
        readme_content += f"- **{dataset['name']}**: {dataset['size']} ({dataset['samples']})\n"
    
    readme_content += f"""
## Total Storage: ~150MB estimated

## Usage Instructions

### Loading Datasets in Python
```python
import pandas as pd
import json

# Load dataset inventory
with open('data/dataset_inventory.json', 'r') as f:
    datasets = json.load(f)

# Load a specific dataset
mmlu_data = pd.read_csv('data/evaluation/mmlu.csv')
print(f"MMLU shape: {{mmlu_data.shape}}")
```

### Dataset Relevance to Inbreeding Analysis

Each dataset serves a specific purpose in measuring LLM quality degradation:

1. **Knowledge Retention**: MMLU, TruthfulQA measure factual accuracy
2. **Reasoning Ability**: GSM8K, MATH, ARC test logical reasoning
3. **Code Quality**: HumanEval, MBPP assess programming capabilities  
4. **Language Understanding**: HellaSwag, WinoGrande test comprehension
5. **Task-Specific Performance**: SuperGLUE tasks provide diverse evaluation

## Experimental Protocol

For inbreeding analysis, these datasets should be used to:
1. Establish baseline performance (Generation 0)
2. Measure degradation after each training iteration
3. Track specific capability losses (reasoning, knowledge, coding)
4. Identify which abilities degrade fastest
5. Validate theoretical predictions about model collapse

## Citation Information

Please cite the original dataset papers when using these benchmarks.
See individual dataset documentation for specific citation requirements.

---
Downloaded on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Total datasets: {len(dataset_info)}
"""
    
    with open('data/README.md', 'w') as f:
        f.write(readme_content)
    
    # Print summary
    print("\n" + "=" * 80)
    print("🎉 DATASET DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"✓ Downloaded {len(dataset_info)} datasets")
    print(f"✓ Created comprehensive documentation")
    print(f"✓ Total estimated size: ~150MB")
    print("\n📁 Dataset structure:")
    for dataset in dataset_info:
        print(f"  • {dataset['name']:15s} → {dataset['path']}")
    
    print(f"\n📖 Complete documentation: data/README.md")
    print(f"📋 Dataset inventory: data/dataset_inventory.json") 
    
    # Verify downloads
    print("\n🔍 Verifying downloads...")
    total_files = 0
    total_size = 0
    
    for dataset in dataset_info:
        if os.path.exists(dataset['path']):
            size = os.path.getsize(dataset['path'])
            total_files += 1
            total_size += size
            print(f"  ✓ {dataset['name']}: {size/1024/1024:.1f}MB")
        else:
            print(f"  ✗ {dataset['name']}: File not found")
    
    print(f"\n📊 Final Summary:")
    print(f"  • Files successfully downloaded: {total_files}/{len(dataset_info)}")
    print(f"  • Total actual size: {total_size/1024/1024:.1f}MB")
    print("  • Ready for LLM inbreeding analysis experiments!")
    
    return dataset_info

if __name__ == "__main__":
    main()