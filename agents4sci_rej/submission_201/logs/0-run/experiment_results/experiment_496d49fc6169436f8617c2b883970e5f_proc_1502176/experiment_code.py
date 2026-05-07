import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from transformers import BertTokenizer, BertModel
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Hyperparameter search space
kernel_sizes = [3, 5, 7]
tuning_type = "cnn_kernel_size"

experiment_data = {tuning_type: {}}  # Each kernel size will be a "dataset" key


# Synthetic claim generator
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
        img_tensor = torch.stack(imgs)  # (3, 1, 28, 28)
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )
        input_ids = enc["input_ids"].squeeze(0)  # (seq_len,)
        attention_mask = enc["attention_mask"].squeeze(0)  # (seq_len,)
        return (
            img_tensor,
            input_ids,
            attention_mask,
            torch.tensor(label, dtype=torch.float32),
        )


# Adaptable CNN for kernel size
class CNNVisionEncoder(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        # Enforce odd kernel size for symmetry and required padding calculation
        assert kernel_size in [3, 5, 7], "Kernel size not supported"
        padding = kernel_size // 2
        # Input: (batch, 3, 28, 28)
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x14x14
            nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),  # 128-dim visual feature
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ClaimVerifier(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.vision = CNNVisionEncoder(kernel_size=kernel_size)
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


def train_eval_loop(model, loaders, optimizer, criterion, num_epochs=10, epoch_start=0):
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []
    last_val_preds, last_val_gts = None, None
    for epoch in range(epoch_start, epoch_start + num_epochs):
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
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
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
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(
            f"Epoch {epoch+1}: train_loss = {tr_loss:.4f}, val_loss = {val_loss:.4f}, train_acc = {tr_acc:.4f}, val_acc = {val_acc:.4f}"
        )
        if epoch == epoch_start + num_epochs - 1:
            last_val_preds = np.concatenate(val_preds)
            last_val_gts = np.concatenate(val_gts)
    return train_accs, val_accs, train_losses, val_losses, last_val_preds, last_val_gts


# Prepare dataset and split _once_ to share across all runs for fair hyperparam tuning
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
full_dataset = MNISTClaimDataset(num_samples=3000, tokenizer=tokenizer)
train_len = int(0.8 * len(full_dataset))
val_len = len(full_dataset) - train_len
train_set, val_set = random_split(
    full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
)

for ks in kernel_sizes:
    dataset_key = f"kernel{ks}x{ks}"
    print(f"\n--- Training with kernel size: {ks}x{ks} ---")
    experiment_data[tuning_type][dataset_key] = {
        "metrics": {"train_acc": [], "val_acc": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
    train_loader = DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    loaders = {"train": train_loader, "val": val_loader}
    model = ClaimVerifier(kernel_size=ks).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    train_accs, val_accs, train_losses, val_losses, val_preds, val_gts = (
        train_eval_loop(model, loaders, optimizer, criterion, num_epochs=10)
    )
    epochs = list(range(1, len(train_accs) + 1))
    experiment_data[tuning_type][dataset_key]["metrics"]["train_acc"] = train_accs
    experiment_data[tuning_type][dataset_key]["metrics"]["val_acc"] = val_accs
    experiment_data[tuning_type][dataset_key]["losses"]["train"] = train_losses
    experiment_data[tuning_type][dataset_key]["losses"]["val"] = val_losses
    experiment_data[tuning_type][dataset_key]["predictions"] = val_preds
    experiment_data[tuning_type][dataset_key]["ground_truth"] = val_gts
    experiment_data[tuning_type][dataset_key]["epochs"] = epochs

    # Visualization for this kernel size
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Train/Validation Accuracy Curve (Kernel {ks}x{ks})")
    plt.legend()
    plot_path = os.path.join(
        working_dir, f"mnist_claims_accuracy_curve_kernel{ks}x{ks}.png"
    )
    plt.savefig(plot_path)
    plt.close()
    print(f"Accuracy curve for kernel size {ks}x{ks} saved to: {plot_path}")
    final_val_acc = val_accs[-1]
    print(f"Final Validation Accuracy (Kernel {ks}x{ks}): {final_val_acc:.4f}")

# Save all experiment data in the required format
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Experiment data saved to: {os.path.join(working_dir, 'experiment_data.npy')}")
