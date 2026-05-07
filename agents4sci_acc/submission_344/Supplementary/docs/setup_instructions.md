# Setup Instructions for SCL Implementation

## System Requirements

### Hardware
- **GPU**: NVIDIA V100 with 32 GB memory (recommended)
- **CPU**: Modern multi-core processor
- **RAM**: 16 GB minimum, 32 GB recommended
- **Storage**: 10 GB free space

### Software
- **Python**: 3.9 or higher
- **CUDA**: 11.0+ (for GPU support)
- **Operating System**: Linux (recommended), macOS, or Windows

## Installation

### 1. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n scl_env python=3.9
conda activate scl_env

# Or using venv
python -m venv scl_env
source scl_env/bin/activate  # Linux/macOS
# scl_env\Scripts\activate  # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')";
python -c "import transformers; print(f'Transformers: {transformers.__version__}')";
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')";
```

## Data Preparation

### Dataset Structure
Create the following directory structure:
```
data/
├── newsnyt/
│   ├── train_human.txt
│   ├── train_ai.txt
│   ├── val_human.txt
│   ├── val_ai.txt
│   ├── test_human.txt
│   └── test_ai.txt
├── argessay/
│   ├── train_human.txt
│   ├── train_ai.txt
│   ├── val_human.txt
│   ├── val_ai.txt
│   ├── test_human.txt
│   └── test_ai.txt
└── chatdialog/
    ├── train_human.txt
    ├── train_ai.txt
    ├── val_human.txt
    ├── val_ai.txt
    ├── test_human.txt
    └── test_ai.txt
```

### Data Format
Each file should contain one text sample per line:
```
File: train_human.txt
I was absolutely thrilled when Sarah told me she'd finally gotten that promotion.
The meeting dragged on forever, and honestly, I was about to fall asleep.
It's funny how things work out sometimes.
```

### Alternative: Using Synthetic Data
The implementation includes synthetic data for demonstration. Set:
```python
USE_SYNTHETIC_DATA = True  # In train_scl_full.py
```

## Model Setup

### 1. Download Pre-trained Models
```bash
# The code will automatically download required models
# roberta-base for style encoder
# bert-base-uncased for tokenizer
```

### 2. Directory Structure
```bash
mkdir -p models outputs data
```

## Training

### Phase 1: Style Encoder Training
```bash
python train_scl_full.py
```

Expected output:
```
Loading datasets...
Starting style encoder training...
Epoch 1/10: Train Loss = 0.2345
Epoch 1/10: Val Loss = 0.1876
...
Checkpoint saved to models/style_encoder_best.pt
```

### Phase 2: Generator Fine-tuning
The generator training requires access to GPT-5 API. Update the code:
```python
# In train_scl_full.py, replace DummyGenerator with:
import openai

class GPT5Generator(nn.Module):
    def __init__(self, api_key):
        super().__init__()
        openai.api_key = api_key

    def forward(self, prompts, style_embedding):
        # Implement GPT-5 API calls with style conditioning
        pass
```

## Evaluation

### Run Evaluation
```bash
python evaluate.py \
    --model_path models/final_model.pt \
    --data_path data/test_data \
    --output outputs/results.csv
```

### Expected Metrics
- **Stylometric Detector Accuracy**: ~50-55% (reduced from baseline 70-75%)
- **Distinct-2**: ~80-85%
- **Idioms per 1k tokens**: ~2.5-3.5
- **Discourse markers per 100 tokens**: ~4.5-5.5

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in SCLConfig
   batch_size: int = 32  # Instead of 64
   ```

2. **Missing Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

3. **Dataset Not Found**
   - Check file paths in `load_data()` function
   - Ensure proper text encoding (UTF-8)

4. **GPT-5 API Issues**
   - Verify API key and quota
   - Check network connectivity
   - Use placeholder generator for testing

### Performance Tips

- Use mixed precision training: `torch.cuda.amp`
- Enable gradient checkpointing for larger models
- Use distributed training for multiple GPUs

## Hardware-Specific Instructions

### For V100 GPU (Recommended)
```bash
export CUDA_VISIBLE_DEVICES=0
python train_scl_full.py
```

### For Multi-GPU Setup
```bash
# Add to SCLConfig
device: str = "cuda:0"  # Specify GPU device
```

### For CPU Only
```bash
# Modify SCLConfig
device: str = "cpu"
batch_size: int = 16  # Reduce batch size
```

## Logging and Monitoring

Training logs are saved to:
- `models/style_encoder_best.pt` - Best style encoder checkpoint
- `models/generator_best.pt` - Best generator checkpoint
- `outputs/training.log` - Detailed training logs

Monitor training progress:
```bash
tail -f outputs/training.log
```

## Reproducing Paper Results

To exactly reproduce the paper results:

1. **Use exact datasets** from reproducibility statement
2. **Match hyperparameters** exactly as specified
3. **Use same hardware** (V100 GPU recommended)
4. **Run evaluation** with same metrics and protocols

Expected training time:
- Style Encoder: ~4 hours
- Generator (per dataset): ~6 hours
- Total: ~22 hours for all three datasets

## Support

For issues or questions:
1. Check this setup guide
2. Review the reproducibility statement
3. Contact the authors through Agents4Science submission system

## Version Information

- **Code Version**: 1.0.0
- **Tested on**: Python 3.9, PyTorch 1.12.1, CUDA 11.6
- **Last Updated**: September 2025
