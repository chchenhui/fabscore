import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from transformers import BertTokenizer, BertModel
import random
import numpy as np
import matplotlib.pyplot as plt

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Experiment data container for batch_size tuning
experiment_data = {
    "batch_size": {
        32: {
            "metrics": {"train_acc": [], "val_acc": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        },
        64: {
            "metrics": {"train_acc": [], "val_acc": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        },
        128: {
            "metrics": {"train_acc": [], "val_acc": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        },
    }
}


def generate_claim(digits):
    claim_type = random.choice(["sum_even", "all_less_than_5"])
    if claim_type == "sum_even":
        label = int(sum(digits) % 2 == 0)
        text = "The sum of the digits is even."
    elif claim_type == "all_less_than_5":
        label = int(all([d < 5 for d in digits]))
        text = "All digits are less than 5."
    return text, label


class MNISTClaimDataset(Dataset):
    def __init__(self, num_samples=3000, tokenizer=None):
        self.data = datasets.MNIST(
            root=".", train=True, download=True, transform=transforms.ToTensor()
        )
        self.num_samples = num_samples
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased")
        self.samples = self._generate()

    def _generate(self):
        samples = []
        for _ in range(self.num_samples):
            indices = random.sample(range(len(self.data)), 3)
            imgs = [self.data[i][0] for i in indices]
            labels = [self.data[i][1] for i in indices]
            text, truth = generate_claim(labels)
            samples.append((imgs, text, truth))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imgs, text, label = self.samples[idx]
        img_tensor = torch.stack(imgs)
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return (
            img_tensor,
            input_ids,
            attention_mask,
            torch.tensor(label, dtype=torch.float32),
        )


class CNNVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ClaimVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = CNNVisionEncoder()
        self.text = BertModel.from_pretrained("bert-base-uncased")
        for param in self.text.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(128 + 768, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, imgs, input_ids, attn_mask):
        vis_feat = self.vision(imgs)
        txt_feat = self.text(
            input_ids=input_ids, attention_mask=attn_mask
        ).last_hidden_state[:, 0, :]
        combined = torch.cat([vis_feat, txt_feat], dim=1)
        out = self.fc(combined).squeeze(1)
        return out


def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    imgs = imgs.squeeze(2)
    input_ids = torch.stack([item[1] for item in batch])
    attn_mask = torch.stack([item[2] for item in batch])
    labels = torch.stack([item[3] for item in batch])
    return imgs, input_ids, attn_mask, labels


def train_eval_loop(
    model, loaders, optimizer, criterion, num_epochs=10, experiment_subdict=None
):
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, n = 0, 0, 0
        for imgs, input_ids, attn_mask, labels in loaders["train"]:
            imgs, input_ids, attn_mask, labels = (
                imgs.to(device),
                input_ids.to(device),
                attn_mask.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            outputs = model(imgs, input_ids, attn_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            n += imgs.size(0)
        tr_loss, tr_acc = total_loss / n, correct / n

        # Validation
        model.eval()
        val_loss, val_correct, val_n = 0, 0, 0
        val_preds, val_gts = [], []
        with torch.no_grad():
            for imgs, input_ids, attn_mask, labels in loaders["val"]:
                imgs, input_ids, attn_mask, labels = (
                    imgs.to(device),
                    input_ids.to(device),
                    attn_mask.to(device),
                    labels.to(device),
                )
                outputs = model(imgs, input_ids, attn_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = (outputs > 0.5).float().cpu().numpy()
                val_preds.append(preds)
                val_gts.append(labels.cpu().numpy())
                val_correct += (preds == labels.cpu().numpy()).sum()
                val_n += imgs.size(0)
        val_loss /= val_n
        val_acc = val_correct / val_n
        print(
            f"Epoch {epoch+1}: train_loss = {tr_loss:.4f}, val_loss = {val_loss:.4f}, train_acc = {tr_acc:.4f}, val_acc = {val_acc:.4f}"
        )
        experiment_subdict["losses"]["train"].append(tr_loss)
        experiment_subdict["losses"]["val"].append(val_loss)
        experiment_subdict["metrics"]["train_acc"].append(tr_acc)
        experiment_subdict["metrics"]["val_acc"].append(val_acc)
        experiment_subdict["epochs"].append(epoch + 1)
        if epoch == num_epochs - 1:
            experiment_subdict["predictions"] = np.concatenate(val_preds)
            experiment_subdict["ground_truth"] = np.concatenate(val_gts)
    return model


# Prepare tokenizer and dataset only once
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
full_dataset = MNISTClaimDataset(num_samples=3000, tokenizer=tokenizer)
train_len = int(0.8 * len(full_dataset))
val_len = len(full_dataset) - train_len
split_indices = list(range(len(full_dataset)))
random.shuffle(split_indices)
train_indices = split_indices[:train_len]
val_indices = split_indices[train_len:]

from torch.utils.data import Subset

train_set = Subset(full_dataset, train_indices)
val_set = Subset(full_dataset, val_indices)

batch_sizes = [32, 64, 128]
colors = {32: "tab:blue", 64: "tab:orange", 128: "tab:green"}
num_epochs = 10

plt.figure(figsize=(8, 5))

for batch_size in batch_sizes:
    print(f"\n=== Training with batch_size={batch_size} ===")
    # Re-create dataloaders for this batch size
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    loaders = {"train": train_loader, "val": val_loader}
    # Model, criterion, optimizer fresh for each run
    model = ClaimVerifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    # Clear data for this batch_size
    subdict = experiment_data["batch_size"][batch_size]
    subdict["metrics"]["train_acc"].clear()
    subdict["metrics"]["val_acc"].clear()
    subdict["losses"]["train"].clear()
    subdict["losses"]["val"].clear()
    subdict["epochs"].clear()
    # Train
    train_eval_loop(
        model,
        loaders,
        optimizer,
        criterion,
        num_epochs=num_epochs,
        experiment_subdict=subdict,
    )
    # Store plot
    plt.plot(
        subdict["epochs"],
        subdict["metrics"]["val_acc"],
        label=f"Val Acc (batch={batch_size})",
        color=colors[batch_size],
        linestyle="-",
    )
    plt.plot(
        subdict["epochs"],
        subdict["metrics"]["train_acc"],
        label=f"Train Acc (batch={batch_size})",
        color=colors[batch_size],
        linestyle="--",
        alpha=0.6,
    )
    # Print final validation accuracy for batch size
    print(
        f"Final val acc (batch_size={batch_size}): {subdict['metrics']['val_acc'][-1]:.4f}"
    )

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train/Validation Accuracy, Varying Batch Size")
plt.legend()
plot_path = os.path.join(working_dir, "mnist_claims_accuracy_curve.png")
plt.savefig(plot_path)
plt.close()
print(f"Accuracy curves saved to: {plot_path}")

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
