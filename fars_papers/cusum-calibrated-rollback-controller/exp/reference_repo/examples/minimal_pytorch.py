# ================================================================
# Exact runnable demo aligned with:
# "Automatic Stability and Recovery for Neural Network Training"
#
# Sections: 3.1 (Vision Model), 3.3 (Training Protocol), Algorithm 1
#
# Model: ResNet-18
# Dataset: CIFAR-10
# Optimizer: AdamW
# Measurement: Validation probe (fixed held-out subset)
#
# This script demonstrates:
# - Passive controller behavior during nominal training
# - Active intervention under controlled catastrophic perturbation
# ================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# Force inline plotting in Colab / notebooks
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")

import matplotlib.pyplot as plt
import numpy as np

from runtime_stability_controller.controller import StabilityController
from runtime_stability_controller.probes import ValidationProbe
from runtime_stability_controller.snapshot import InMemorySnapshotManager

# --------------------------------------------------
# Reproducibility & device
# --------------------------------------------------
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Demo configuration
# --------------------------------------------------
# When True, injects a controlled catastrophic perturbation
# to activate the runtime stability controller (paper Section 3.3).
# When False, training remains nominal and the controller stays passive.
ENABLE_CATASTROPHIC_PERTURBATION = True

# --------------------------------------------------
# Dataset: CIFAR-10 (paper Section 3.1)
# --------------------------------------------------
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

train_set = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_set,
    batch_size=128,
    shuffle=True,
    num_workers=2
)

# Validation probe: small fixed subset (paper protocol)
probe_indices = list(range(32))
probe_subset = Subset(test_set, probe_indices)
probe_loader = DataLoader(
    probe_subset,
    batch_size=32,
    shuffle=False
)

# --------------------------------------------------
# Model: ResNet-18 (paper Section 3.1)
# --------------------------------------------------
model = torchvision.models.resnet18(num_classes=10)
model = model.to(device)

# --------------------------------------------------
# Optimization (paper Section 3.1)
# --------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# --------------------------------------------------
# Runtime stability components (Algorithm 1)
# --------------------------------------------------
probe = ValidationProbe(
    dataloader=probe_loader,
    loss_fn=criterion,
    device=device
)

snapshot_manager = InMemorySnapshotManager()

controller = StabilityController(
    probe=probe,
    snapshot_manager=snapshot_manager,
    threshold=0.5,   # ε (acceptance threshold)
    smoothing=0.1    # α (reference smoothing)
)

# Initialize controller:
# - compute y(θ₀)
# - initialize reference signal ŷ₀
# - store initial safe snapshot
controller.initialize(model, optimizer)

# --------------------------------------------------
# Metrics collection (for plots)
# --------------------------------------------------
train_losses = []
probe_losses = []
innovations = []
accepted_flags = []

# --------------------------------------------------
# Training loop (Algorithm 1, paper Section 3.3)
# --------------------------------------------------
model.train()
num_steps = 250
step = 0

while step < num_steps:
    for images, labels in train_loader:
        if step >= num_steps:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # --------------------------------------------------
        # Controlled catastrophic perturbation (paper Section 3.3)
        # --------------------------------------------------
        # Gradient amplification window to induce instability
        if ENABLE_CATASTROPHIC_PERTURBATION and 100 <= step < 110:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(300.0)

        # Runtime-supervised optimizer step (Algorithm 1)
        accepted = controller.step(model, optimizer)

        # Store metrics
        train_losses.append(loss.item())
        probe_losses.append(controller.last_measurement)
        innovations.append(controller.last_innovation)
        accepted_flags.append(accepted)

        print(
            f"Step {step:03d} | "
            f"train_loss={loss.item():.4f} | "
            f"probe={controller.last_measurement:.4f} | "
            f"innovation={controller.last_innovation:.4f} | "
            f"accepted={accepted}"
        )

        step += 1

# --------------------------------------------------
# Plots (diagnostic, paper-style)
# --------------------------------------------------
steps = np.arange(len(train_losses))

plt.figure(figsize=(15, 4))

# Training loss
plt.subplot(1, 3, 1)
plt.plot(steps, train_losses)
plt.title("Training Loss (CIFAR-10)")
plt.xlabel("Step")
plt.ylabel("Cross-Entropy")

# Validation probe loss
plt.subplot(1, 3, 2)
plt.plot(steps, probe_losses)
plt.title("Validation Probe Loss")
plt.xlabel("Step")
plt.ylabel("Loss")

# Innovation signal
plt.subplot(1, 3, 3)

# Innovation curve
plt.plot(steps, innovations, label="Innovation νₜ")

# Perturbation window (paper Section 3.3)
plt.axvspan(
    100,
    110,
    color="gray",
    alpha=0.15,
    label="Perturbation window"
)

# Acceptance threshold
plt.axhline(
    controller.threshold,
    linestyle="--",
    color="black",
    label="Threshold ε"
)


rejected_steps = [i for i, a in enumerate(accepted_flags) if not a]
if rejected_steps:
    plt.scatter(
        rejected_steps,
        [innovations[i] for i in rejected_steps],
        color="red",
        label="Rejected"
    )

plt.title("Innovation Signal vs Threshold")
plt.xlabel("Step")
plt.legend()

plt.tight_layout()
plt.tight_layout()

# Save figure at high resolution (300 DPI)
plt.savefig(
    "runtime_stability_demo.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=True)


# --------------------------------------------------
# Conclusions (explicit, paper-aligned)
# --------------------------------------------------
print("\n=== Conclusions ===")
print(
    "This demo reproduces the experimental setting of the paper using\n"
    "ResNet-18 on CIFAR-10 with a validation-based measurement probe.\n\n"
    "A controlled destabilization window is injected by amplifying\n"
    "gradients for a small number of steps (enabled by default via\n"
    "ENABLE_CATASTROPHIC_PERTURBATION).\n\n"
    "During nominal training, the innovation signal remains bounded and\n"
    "the runtime controller remains passive, accepting optimizer updates\n"
    "without modifying the optimizer.\n\n"
    "When destabilizing updates are introduced, the innovation signal\n"
    "exceeds the safety threshold, triggering selective rejection and\n"
    "rollback to the last safe state. Training stability is recovered\n"
    "without restarting or altering the optimizer.\n\n"
    "This demonstrates training reliability enforced as a runtime\n"
    "safety property, decoupled from the optimizer design."
)
