# Verify CBraMod model loads and runs a forward pass on GPU.
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "CBraMod"))

import torch
import torch.nn as nn
from models.cbramod import CBraMod

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = CBraMod().to(device)
weights_path = os.path.join(PROJECT_ROOT, "third_party", "CBraMod", "pretrained_weights", "pretrained_weights.pth")
state_dict = torch.load(weights_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")

model.proj_out = nn.Identity()

mock_eeg = torch.randn((4, 22, 4, 200)).to(device)
with torch.no_grad():
    out = model(mock_eeg)
print(f"Input shape: {mock_eeg.shape}")
print(f"Output shape: {out.shape}")
print(f"Output dtype: {out.dtype}")
print(f"Output mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")

assert out.shape == (4, 22, 4, 200), f"Unexpected output shape: {out.shape}"
print("CBraMod verification PASSED")
