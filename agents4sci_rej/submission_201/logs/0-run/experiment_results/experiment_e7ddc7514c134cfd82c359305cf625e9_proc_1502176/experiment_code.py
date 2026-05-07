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

# Set up experiment directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Experiment data container
experiment_data = {
    "learning_rate_tuning": {},
}


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
        vis_feat = self.vision(imgs)  # (batch,128)
        txt_feat = self.text(
            input_ids=input_ids, attention_mask=attn_mask
        ).last_hidden_state[
            :, 0, :
        ]  # (batch,768)
        combined = torch.cat([vis_feat, txt_feat], dim=1)
        out = self.fc(combined).squeeze(1)
        return out


def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])  # (B, 3, 1, 28, 28)
    imgs = imgs.squeeze(2)
    input_ids = torch.stack([item[1] for item in batch])
    attn_mask = torch.stack([item[2] for item in batch])
    labels = torch.stack([item[3] for item in batch])
    return imgs, input_ids, attn_mask, labels


def train_eval_loop(model, loaders, optimizer, criterion, num_epochs=10, epoch_start=0):
    losses = {"train": [], "val": []}
    metrics = {"train": [], "val": []}
    val_preds_final, val_gts_final = None, None

    for epoch in range(epoch_start, epoch_start + num_epochs):
        model.train()
        total_loss, correct, n = 0, 0, 0
        for imgs, input_ids, attn_mask, labels in loaders["train"]:
            imgs = imgs.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, input_ids, attn_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            n += imgs.size(0)
        tr_loss = total_loss / n
        tr_acc = correct / n

        # Validation
        model.eval()
        val_loss, val_correct, val_n = 0, 0, 0
        val_preds, val_gts = [], []
        with torch.no_grad():
            for imgs, input_ids, attn_mask, labels in loaders["val"]:
                imgs = imgs.to(device)
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device)
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
        losses["train"].append(tr_loss)
        losses["val"].append(val_loss)
        metrics["train"].append(tr_acc)
        metrics["val"].append(val_acc)
        if epoch == epoch_start + num_epochs - 1:
            val_preds_final = np.concatenate(val_preds)
            val_gts_final = np.concatenate(val_gts)
        print(
            f"Epoch {epoch+1}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} train_acc={tr_acc:.4f} val_acc={val_acc:.4f}"
        )
    return losses, metrics, val_preds_final, val_gts_final


########################################################################
# Prepare dataset and data loaders once, reuse for all runs
print("Preparing dataset and dataloaders...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
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
########################################################################

# Learning rates to try
learning_rates = [5e-5, 1e-4, 5e-4]
num_epochs = 10

for lr in learning_rates:
    lr_key = f"lr_{lr:.0e}".replace("+0", "")
    experiment_data["learning_rate_tuning"][lr_key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": list(range(1, num_epochs + 1)),
    }
    # Reset model and optimizer for each run
    print(f"\nTraining with learning rate = {lr}")
    model = ClaimVerifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # Train/eval
    losses, metrics, val_preds, val_gts = train_eval_loop(
        model, loaders, optimizer, criterion, num_epochs=num_epochs
    )
    experiment_data["learning_rate_tuning"][lr_key]["metrics"]["train"] = metrics[
        "train"
    ]
    experiment_data["learning_rate_tuning"][lr_key]["metrics"]["val"] = metrics["val"]
    experiment_data["learning_rate_tuning"][lr_key]["losses"]["train"] = losses["train"]
    experiment_data["learning_rate_tuning"][lr_key]["losses"]["val"] = losses["val"]
    experiment_data["learning_rate_tuning"][lr_key]["predictions"] = val_preds
    experiment_data["learning_rate_tuning"][lr_key]["ground_truth"] = val_gts

    # Plot for this run
    plt.figure(figsize=(8, 5))
    plt.plot(
        experiment_data["learning_rate_tuning"][lr_key]["epochs"],
        metrics["train"],
        label="Train Accuracy",
    )
    plt.plot(
        experiment_data["learning_rate_tuning"][lr_key]["epochs"],
        metrics["val"],
        label="Validation Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve (lr={lr})")
    plt.legend()
    plot_path = os.path.join(
        working_dir, f"acc_curve_lr_{lr:.0e}.png".replace("+0", "")
    )
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved accuracy curve to {plot_path}")

# Multi-run comparison plot
plt.figure(figsize=(8, 5))
for lr in learning_rates:
    lr_key = f"lr_{lr:.0e}".replace("+0", "")
    plt.plot(
        experiment_data["learning_rate_tuning"][lr_key]["epochs"],
        experiment_data["learning_rate_tuning"][lr_key]["metrics"]["val"],
        label=f"Val acc lr={lr}",
    )
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs Epochs (Learning Rate Tuning)")
plt.legend()
comp_path = os.path.join(working_dir, "acc_curve_lr_compare.png")
plt.savefig(comp_path)
plt.close()
print(f"Comparison curve saved to: {comp_path}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Print final best val accuracy for all runs
for lr in learning_rates:
    lr_key = f"lr_{lr:.0e}".replace("+0", "")
    final_val_acc = experiment_data["learning_rate_tuning"][lr_key]["metrics"]["val"][
        -1
    ]
    print(f"Final Validation Accuracy for lr={lr}: {final_val_acc:.4f}")
