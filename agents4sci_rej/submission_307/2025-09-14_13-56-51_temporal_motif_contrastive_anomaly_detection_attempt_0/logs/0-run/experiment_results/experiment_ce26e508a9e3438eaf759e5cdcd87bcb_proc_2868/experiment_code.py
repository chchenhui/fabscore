import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experiment data storage
experiment_data = {"batch_size_tuning": {}}


# Simple synthetic data generation
def generate_synthetic_data(
    num_samples=10, num_nodes=100, num_features=16, num_classes=2
):
    data_list = []
    for _ in range(num_samples):
        x = torch.randn((num_nodes, num_features), dtype=torch.float)
        edge_index = (
            torch.tensor(
                [[i, (i + 1) % num_nodes] for i in range(num_nodes)], dtype=torch.long
            )
            .t()
            .contiguous()
        )
        y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index, y=y))
    return data_list


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
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


# Evaluation function
def evaluate(model, loader):
    model.eval()
    total_f1 = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            f1 = f1_score(batch.y.cpu(), pred.cpu(), average="weighted")
            total_f1 += f1 * batch.num_graphs
    return total_f1 / len(loader.dataset)


# Generate data and model
data_list = generate_synthetic_data()
loader = DataLoader(data_list, batch_size=32, shuffle=True)
model = SimpleGNN(num_features=16, num_classes=2).to(device)

# Hyperparameter tuning for batch sizes
batch_sizes = [8, 16, 32, 64]
epochs = 10

for batch_size in batch_sizes:
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss = train(model, loader, optimizer)
        val_f1 = evaluate(model, loader)

        # Track and save metrics and losses
        experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"]["losses"][
            "train"
        ].append(train_loss)
        experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"]["metrics"][
            "val"
        ].append(val_f1)

        print(
            f"Batch size {batch_size}, Epoch {epoch}: train_loss = {train_loss:.4f}, val_f1 = {val_f1:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
