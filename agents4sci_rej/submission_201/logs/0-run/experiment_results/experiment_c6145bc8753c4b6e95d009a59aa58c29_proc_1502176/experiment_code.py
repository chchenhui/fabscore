import os
import torch

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Hyperparameter Tuning Container ---
experiment_data = {
    "optimizer_type": {
        "mnist_claims": {
            # keys will be optimizer names, values as dict with metrics etc.
        }
    }
}


# --- Synthetic claim generator ---
def generate_claim(digits):
    claim_type = random.choice(["sum_even", "all_less_than_5"])
    if claim_type == "sum_even":
        label = int(sum(digits) % 2 == 0)
        text = "The sum of the digits is even."
    elif claim_type == "all_less_than_5":
        label = int(all([d < 5 for d in digits]))
        text = "All digits are less than 5."
    return text, label


# --- Custom MNIST+Claim dataset ---
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
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return (
            img_tensor,
            input_ids,
            attention_mask,
            torch.tensor(label, dtype=torch.float32),
        )


# --- Simple CNN for vision encoding ---
class CNNVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 3->16
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x14x14
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# --- Full multi-modal claim verifier model ---
class ClaimVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = CNNVisionEncoder()
        self.text = BertModel.from_pretrained("bert-base-uncased")
        for param in self.text.parameters():
            param.requires_grad = False  # freeze BERT
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
    imgs = torch.stack([item[0] for item in batch]).squeeze(2)
    input_ids = torch.stack([item[1] for item in batch])
    attn_mask = torch.stack([item[2] for item in batch])
    labels = torch.stack([item[3] for item in batch])
    return imgs, input_ids, attn_mask, labels


# --- Training and validation loop (single run) ---
def train_eval_loop(model, loaders, optimizer, criterion, num_epochs=10):
    run_metrics = {"train_acc": [], "val_acc": []}
    run_losses = {"train": [], "val": []}
    predictions = []
    ground_truth = []
    epochs = []
    best_val_acc = 0.0
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
        v_loss = val_loss / val_n
        v_acc = val_correct / val_n

        run_losses["train"].append(tr_loss)
        run_losses["val"].append(v_loss)
        run_metrics["train_acc"].append(tr_acc)
        run_metrics["val_acc"].append(v_acc)
        epochs.append(epoch + 1)
        print(
            f"Epoch {epoch+1}: train_loss={tr_loss:.4f}, val_loss={v_loss:.4f}, train_acc={tr_acc:.4f}, val_acc={v_acc:.4f}"
        )

        # Save predictions/ground truths only at last epoch
        if epoch == num_epochs - 1:
            predictions = np.concatenate(val_preds)
            ground_truth = np.concatenate(val_gts)
    return run_metrics, run_losses, epochs, predictions, ground_truth


# --- Data Preparation (run only once) ---
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

# --- Optimizer hyperparameter search setup ---
optimizer_hyperparams = [
    (
        "adam",
        lambda model: optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
        ),
    ),
    (
        "sgd",
        lambda model: optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2
        ),
    ),
    (
        "sgd_momentum",
        lambda model: optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2, momentum=0.9
        ),
    ),
    (
        "rmsprop",
        lambda model: optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
        ),
    ),
]

num_epochs = 10
criterion = nn.BCELoss()

for opt_name, opt_fn in optimizer_hyperparams:
    print(f"\n=========== Training with optimizer: {opt_name} ===========")
    # New model per optimizer trial
    model = ClaimVerifier().to(device)
    optimizer = opt_fn(model)
    run_metrics, run_losses, epochs, preds, gts = train_eval_loop(
        model, loaders, optimizer, criterion, num_epochs=num_epochs
    )
    # Record experiment data
    experiment_data["optimizer_type"]["mnist_claims"][opt_name] = {
        "metrics": {
            "train_acc": run_metrics["train_acc"],
            "val_acc": run_metrics["val_acc"],
        },
        "losses": {"train": run_losses["train"], "val": run_losses["val"]},
        "predictions": preds,
        "ground_truth": gts,
        "epochs": epochs,
    }

    # Save figure for this optimizer
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, run_metrics["train_acc"], label="Train Accuracy")
    plt.plot(epochs, run_metrics["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Train/Validation Accuracy Curve ({opt_name})")
    plt.legend()
    plot_path = os.path.join(working_dir, f"mnist_claims_accuracy_curve_{opt_name}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Accuracy curve for {opt_name} saved to: {plot_path}")
    # Save after each run for safety (final overwrite at end)
    np.save(
        os.path.join(working_dir, "experiment_data.npy"),
        experiment_data,
        allow_pickle=True,
    )

# (Optional) Overlay plot for all optimizer runs
plt.figure(figsize=(8, 5))
for opt_name in experiment_data["optimizer_type"]["mnist_claims"]:
    plt.plot(
        experiment_data["optimizer_type"]["mnist_claims"][opt_name]["epochs"],
        experiment_data["optimizer_type"]["mnist_claims"][opt_name]["metrics"][
            "val_acc"
        ],
        label=f"{opt_name} val_acc",
    )
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Curve (All Optimizers)")
plt.legend()
overall_plot_path = os.path.join(
    working_dir, "mnist_claims_accuracy_curve_all_optimizers.png"
)
plt.savefig(overall_plot_path)
plt.close()
print(f"Overlay accuracy curve saved to: {overall_plot_path}")

# Final save of full experiment data
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)

# Print final validation accuracy per optimizer
for opt_name, d in experiment_data["optimizer_type"]["mnist_claims"].items():
    final_val_acc = d["metrics"]["val_acc"][-1]
    print(f"Final Validation Accuracy ({opt_name}): {final_val_acc:.4f}")
