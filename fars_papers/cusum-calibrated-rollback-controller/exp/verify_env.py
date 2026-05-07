# Quick sanity check to verify the environment is correctly set up.
# Tests: torch CUDA, torchvision ResNet-18, CIFAR-10 loading, scipy, matplotlib, seaborn,
# and the reference_repo import path.

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reference_repo"))

print("=== Environment Verification ===")

import torch
print(f"[OK] torch {torch.__version__}")
print(f"[OK] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[OK] CUDA device: {torch.cuda.get_device_name(0)}")
    x = torch.randn(2, 3, device="cuda")
    y = x @ x.T
    print(f"[OK] CUDA tensor ops work: {y.shape}")

import torchvision
print(f"[OK] torchvision {torchvision.__version__}")

model = torchvision.models.resnet18(num_classes=10)
model.eval()
dummy = torch.randn(2, 3, 32, 32)
out = model(dummy)
assert out.shape == (2, 10), f"Unexpected output shape: {out.shape}"
print(f"[OK] ResNet-18 forward pass: input (2,3,32,32) -> output {out.shape}")

if torch.cuda.is_available():
    model_gpu = model.cuda()
    dummy_gpu = dummy.cuda()
    out_gpu = model_gpu(dummy_gpu)
    assert out_gpu.shape == (2, 10)
    print(f"[OK] ResNet-18 GPU forward pass works")
    model_gpu.train()
    dummy_batch = torch.randn(16, 3, 32, 32, device="cuda")
    out_train = model_gpu(dummy_batch)
    assert out_train.shape == (16, 10)
    print(f"[OK] ResNet-18 GPU training-mode forward pass works")

import torchvision.transforms as T
from torch.utils.data import DataLoader
transform = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
try:
    ds = torchvision.datasets.CIFAR10(root="./data_verify_tmp", train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    imgs, labels = next(iter(loader))
    assert imgs.shape == (4, 3, 32, 32)
    print(f"[OK] CIFAR-10 loaded: batch shape {imgs.shape}, labels {labels.tolist()}")
except Exception as e:
    print(f"[WARN] CIFAR-10 download issue (may need internet): {e}")

import numpy as np
print(f"[OK] numpy {np.__version__}")
arr = np.random.randn(100)
print(f"[OK] numpy random array mean={arr.mean():.4f}")

import scipy
from scipy import stats
print(f"[OK] scipy {scipy.__version__}")
ks_stat, p_val = stats.kstest(arr, 'norm')
print(f"[OK] scipy.stats KS test: stat={ks_stat:.4f}, p={p_val:.4f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
print(f"[OK] matplotlib {matplotlib.__version__}")

import seaborn as sns
print(f"[OK] seaborn {sns.__version__}")

import pandas as pd
print(f"[OK] pandas {pd.__version__}")

from tqdm import tqdm
print(f"[OK] tqdm available")

import tensorboard
print(f"[OK] tensorboard {tensorboard.__version__}")

from runtime_stability_controller.controller import StabilityController
from runtime_stability_controller.probes import ValidationProbe
from runtime_stability_controller.snapshot import InMemorySnapshotManager
print(f"[OK] reference_repo imports: StabilityController, ValidationProbe, InMemorySnapshotManager")

import yaml
print(f"[OK] PyYAML available")

print("\n=== All checks passed ===")
