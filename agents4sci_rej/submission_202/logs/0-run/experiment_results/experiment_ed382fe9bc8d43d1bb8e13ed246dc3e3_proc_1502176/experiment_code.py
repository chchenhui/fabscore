import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import random
import numpy as np
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {"num_conv_layers": {}}


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


class CNNVisionEncoder(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        layers = []
        in_channels = 3
        # Shared configuration for all layers
        conv_cfgs = [
            (16, 3, 1),  # out_channels, kernel_size, stride/padding
            (32, 3, 1),
            (64, 3, 1),
        ]
        n_conv_cfg = min(num_layers, len(conv_cfgs))
        for i in range(num_layers):
            out_ch = conv_cfgs[i][0] if i < len(conv_cfgs) else 64
            layers.append(
                nn.Conv2d(in_channels, out_ch, kernel_size=3, stride=1, padding=1)
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_ch
        self.conv = nn.Sequential(*layers)
        # For spatial size after convs+pooling, compute size
        size = 28
        for _ in range(num_layers):
            size = size // 2
        out_feat_dim = in_channels * size * size
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_feat_dim, 128),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.final(x)
        return x


class ClaimVerifier(nn.Module):
    def __init__(self, num_conv_layers=2):
        super().__init__()
        self.vision = CNNVisionEncoder(num_layers=num_conv_layers)
        self.text = BertModel.from_pretrained("bert-base-uncased")
        for param in self.text.parameters():
            param.requires_grad = False  # freeze BERT for baseline
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
    imgs = torch.stack([item[0] for item in batch])  # (B, 3, 1, 28, 28)
    imgs = imgs.squeeze(2)  # (B, 3, 28, 28)
    input_ids = torch.stack([item[1] for item in batch])
    attn_mask = torch.stack([item[2] for item in batch])
    labels = torch.stack([item[3] for item in batch])
    return imgs, input_ids, attn_mask, labels


def train_eval_loop(
    model, loaders, optimizer, criterion, num_epochs=10, epoch_start=0, exp_dict=None
):
    best_val_acc = 0.0
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
        exp_dict["losses"]["train"].append(tr_loss)
        exp_dict["losses"]["val"].append(val_loss)
        exp_dict["metrics"]["train"].append(tr_acc)
        exp_dict["metrics"]["val"].append(val_acc)
        exp_dict["epochs"].append(epoch + 1)
        if epoch == epoch_start + num_epochs - 1:
            exp_dict["predictions"] = np.concatenate(val_preds)
            exp_dict["ground_truth"] = np.concatenate(val_gts)
    return model


# HYPERPARAM TUNING: try with 1, 2, 3 conv layers
num_layer_options = [1, 2, 3]
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
for n_layers in num_layer_options:
    exp_key = f"ch_{n_layers}_layers"
    # Prepare new experiment dict
    experiment_data["num_conv_layers"][exp_key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
        "n_layers": n_layers,
    }
    print(f"\n--- Tuning number of conv layers: {n_layers} ---")
    # Fix data split for comparability
    full_dataset = MNISTClaimDataset(num_samples=3000, tokenizer=tokenizer)
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_set, val_set = random_split(
        full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )
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
    # Model, criterion, optimizer
    model = ClaimVerifier(num_conv_layers=n_layers).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    # Training
    train_eval_loop(
        model,
        loaders,
        optimizer,
        criterion,
        num_epochs=10,
        exp_dict=experiment_data["num_conv_layers"][exp_key],
    )

# Save all result data in required format
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Visualization
plt.figure(figsize=(10, 6))
for n_layers in num_layer_options:
    exp_key = f"ch_{n_layers}_layers"
    epochs = experiment_data["num_conv_layers"][exp_key]["epochs"]
    val_acc = experiment_data["num_conv_layers"][exp_key]["metrics"]["val"]
    train_acc = experiment_data["num_conv_layers"][exp_key]["metrics"]["train"]
    plt.plot(epochs, train_acc, label=f"Train ({n_layers} conv)")
    plt.plot(epochs, val_acc, label=f"Val ({n_layers} conv)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Effect of CNN Convolutional Layers on Claim Verification")
plt.legend()
plot_path = os.path.join(working_dir, "mnist_claims_num_conv_layers_accuracy_curve.png")
plt.savefig(plot_path)
plt.close()
print(f"Accuracy curve saved to: {plot_path}")

# Print summary
for n_layers in num_layer_options:
    exp_key = f"ch_{n_layers}_layers"
    final_val_acc = experiment_data["num_conv_layers"][exp_key]["metrics"]["val"][-1]
    print(f"Final Validation Accuracy ({n_layers} conv layers): {final_val_acc:.4f}")
