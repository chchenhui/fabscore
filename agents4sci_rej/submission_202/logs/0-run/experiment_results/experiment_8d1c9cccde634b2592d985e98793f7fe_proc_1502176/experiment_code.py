# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

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

# Set a random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Experiment data container (for hyperparam tuning: num_epochs)
experiment_data = {
    "num_epochs_tuning": {
        "mnist_claims": {
            # keys are epoch counts, will fill below
        }
    }
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


# Custom MNIST+Claim dataset
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


# Simple CNN for processing stack of 3 images as 3 channels
class CNNVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 3->16, 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x14x14
            nn.Conv2d(16, 32, 3, padding=1),  # 32x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),  # 128-dim visual feature
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# Full claim verifier model
class ClaimVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = CNNVisionEncoder()
        self.text = BertModel.from_pretrained("bert-base-uncased")
        for param in self.text.parameters():
            param.requires_grad = False  # freeze BERT for baseline
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
        combined = torch.cat([vis_feat, txt_feat], dim=1)  # (batch,896)
        out = self.fc(combined).squeeze(1)
        return out


def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])  # (B, 3, 1, 28, 28)
    imgs = imgs.squeeze(2)  # (B, 3, 28, 28)
    input_ids = torch.stack([item[1] for item in batch])  # (B, seq)
    attn_mask = torch.stack([item[2] for item in batch])  # (B, seq)
    labels = torch.stack([item[3] for item in batch])  # (B,)
    return imgs, input_ids, attn_mask, labels


# Training and validation loop
def train_eval_loop(model, loaders, optimizer, criterion, num_epochs=10, epoch_start=0):
    (
        train_accs,
        val_accs,
        train_losses,
        val_losses,
        all_val_preds,
        all_val_gts,
        all_epochs,
    ) = ([], [], [], [], None, None, [])
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
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)
        all_epochs.append(epoch + 1)
        # Save preds/gts from final epoch
        if epoch == epoch_start + num_epochs - 1:
            all_val_preds = np.concatenate(val_preds)
            all_val_gts = np.concatenate(val_gts)
    return {
        "metrics": {"train_acc": train_accs, "val_acc": val_accs},
        "losses": {"train": train_losses, "val": val_losses},
        "predictions": all_val_preds,
        "ground_truth": all_val_gts,
        "epochs": all_epochs,
    }


# Prepare dataset, train/val split, and dataloaders (done only once)
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

# Hyperparameter tuning on num_epochs
epoch_options = [10, 20, 30]
colors = ["b", "g", "r"]
plt.figure(figsize=(9, 6))

for idx, num_epochs in enumerate(epoch_options):
    print(f"\n=== Training with num_epochs={num_epochs} ===")
    # Re-initialize model and optimizer each time
    model = ClaimVerifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    # Train and eval
    result = train_eval_loop(
        model, loaders, optimizer, criterion, num_epochs=num_epochs
    )
    # Collect in experiment_data under current num_epochs
    experiment_data["num_epochs_tuning"]["mnist_claims"][
        f"epochs_{num_epochs}"
    ] = result
    # Plot accuracy curve for this setting
    plt.plot(
        result["epochs"],
        result["metrics"]["train_acc"],
        linestyle="--",
        color=colors[idx],
        alpha=0.6,
        label=f"Train Acc (epochs={num_epochs})",
    )
    plt.plot(
        result["epochs"],
        result["metrics"]["val_acc"],
        linestyle="-",
        color=colors[idx],
        label=f"Val Acc (epochs={num_epochs})",
    )

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train/Validation Accuracy Curves (num_epochs tuning)")
plt.legend()
plot_path = os.path.join(working_dir, "mnist_claims_accuracy_curve.png")
plt.savefig(plot_path)
plt.close()
print(f"\nAccuracy curves for all epoch settings saved to: {plot_path}")

# Save experiment data as required
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Print final val accuracy for each setting
for num_epochs in epoch_options:
    acc = experiment_data["num_epochs_tuning"]["mnist_claims"][f"epochs_{num_epochs}"][
        "metrics"
    ]["val_acc"][-1]
    print(f"Final Validation Accuracy (num_epochs={num_epochs}): {acc:.4f}")
