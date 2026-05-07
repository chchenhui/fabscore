# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experiment data storage
experiment_data = {
    "synthetic_dynamic_network": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# Simple synthetic data generation
def generate_synthetic_data(num_nodes=100, num_features=16, num_classes=2):
    x = torch.randn((num_nodes, num_features), dtype=torch.float)
    edge_index = (
        torch.tensor(
            [[i, (i + 1) % num_nodes] for i in range(num_nodes)], dtype=torch.long
        )
        .t()
        .contiguous()
    )
    y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


# Basic GNN model
class SimpleGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Training function
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()


# Evaluation function
def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        f1 = f1_score(data.y.cpu(), pred.cpu(), average="weighted")
    return f1, pred.cpu()


# Generate data and model
data = generate_synthetic_data().to(device)
model = SimpleGNN(num_features=16, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 10
for epoch in range(1, epochs + 1):
    train_loss = train(model, data, optimizer)
    val_f1, predictions = evaluate(model, data)

    # Track and save metrics and losses
    experiment_data["synthetic_dynamic_network"]["losses"]["train"].append(train_loss)
    experiment_data["synthetic_dynamic_network"]["metrics"]["val"].append(val_f1)
    experiment_data["synthetic_dynamic_network"]["predictions"] = (
        predictions.cpu().numpy().tolist()
    )
    experiment_data["synthetic_dynamic_network"]["ground_truth"] = (
        data.y.cpu().numpy().tolist()
    )

    print(f"Epoch {epoch}: train_loss = {train_loss:.4f}, val_f1 = {val_f1:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
