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
    "node_count_ablation": {
        "synthetic_dynamic_network": {
            "metrics": [],
            "losses": [],
            "predictions": [],
            "ground_truth": [],
            "node_count_settings": [],
            "temporal_motif_coverage": [],
        }
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


# Function to calculate Temporal Motif Coverage (TMC)
def calculate_tmc(data):
    # This is a placeholder implementation; replace with actual motif extraction logic.
    return np.random.rand()  # Random value as a stand-in for actual TMC


# Different node count settings to evaluate
node_counts = [50, 150, 200]
epochs = 20  # Fixed epoch setting for ablation

for num_nodes in node_counts:
    data = generate_synthetic_data(num_nodes=num_nodes).to(device)
    model = SimpleGNN(num_features=16, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    node_metrics = {"train": [], "val": []}
    node_losses = {"train": [], "val": []}
    node_predictions = []
    node_ground_truth = []
    node_tmc = []

    for epoch in range(1, epochs + 1):
        train_loss = train(model, data, optimizer)
        val_f1, predictions = evaluate(model, data)
        val_loss = F.nll_loss(model(data), data.y).item()
        tmc = calculate_tmc(data)

        # Track and save metrics, losses, and TMC
        node_losses["train"].append(train_loss)
        node_losses["val"].append(val_loss)
        node_metrics["val"].append(val_f1)
        node_predictions.append(predictions.cpu().numpy().tolist())
        node_ground_truth.append(data.y.cpu().numpy().tolist())
        node_tmc.append(tmc)

        print(
            f"Node Count {num_nodes}: Epoch {epoch}/{epochs}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}, val_f1 = {val_f1:.4f}, TMC = {tmc:.4f}"
        )

    experiment_data["node_count_ablation"]["synthetic_dynamic_network"][
        "metrics"
    ].append(node_metrics)
    experiment_data["node_count_ablation"]["synthetic_dynamic_network"][
        "losses"
    ].append(node_losses)
    experiment_data["node_count_ablation"]["synthetic_dynamic_network"][
        "predictions"
    ].append(node_predictions)
    experiment_data["node_count_ablation"]["synthetic_dynamic_network"][
        "ground_truth"
    ].append(node_ground_truth)
    experiment_data["node_count_ablation"]["synthetic_dynamic_network"][
        "node_count_settings"
    ].append(num_nodes)
    experiment_data["node_count_ablation"]["synthetic_dynamic_network"][
        "temporal_motif_coverage"
    ].append(node_tmc)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
