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

# Experiment data container
experiment_data = {"activation_fn_tuning": {}}


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


# Helper: get activation module from name
def get_activation(activation_name):
    name = activation_name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    elif name == "elu":
        return nn.ELU()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {activation_name}")


# Generalized CNNVisionEncoder
class CNNVisionEncoder(nn.Module):
    def __init__(self, activation_fn_name="relu"):
        super().__init__()
        act = get_activation(activation_fn_name)
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 3->16, 28x28
            act,
            nn.MaxPool2d(2),  # 16x14x14
            nn.Conv2d(16, 32, 3, padding=1),  # 32x14x14
            act,
            nn.MaxPool2d(2),  # 32x7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),  # 128-dim visual feature
            act,
        )

    def forward(self, x):
        return self.net(x)


# Full claim verifier model
class ClaimVerifier(nn.Module):
    def __init__(self, activation_fn_name="relu"):
        super().__init__()
        self.vision = CNNVisionEncoder(activation_fn_name=activation_fn_name)
        self.text = BertModel.from_pretrained("bert-base-uncased")
        for param in self.text.parameters():
            param.requires_grad = False  # freeze BERT for baseline
        # For fairness, keep activation fn in fc always ReLU
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
def train_eval_loop(
    model,
    loaders,
    optimizer,
    criterion,
    num_epochs=10,
    epoch_start=0,
    acc_metric_key="train_acc",
    val_metric_key="val_acc",
    activation_key=None,
):
    best_val_acc = 0.0
    tr_acc_hist, val_acc_hist = [], []
    tr_loss_hist, val_loss_hist = [], []
    val_preds_hist, val_gts_hist = [], []
    epochs_hist = []
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
            f"[{activation_key}] Epoch {epoch+1}: train_loss = {tr_loss:.4f}, val_loss = {val_loss:.4f}, train_acc = {tr_acc:.4f}, val_acc = {val_acc:.4f}"
        )
        tr_loss_hist.append(tr_loss)
        val_loss_hist.append(val_loss)
        tr_acc_hist.append(tr_acc)
        val_acc_hist.append(val_acc)
        epochs_hist.append(epoch + 1)
        # Save last epoch preds/gts for analysis
        if epoch == epoch_start + num_epochs - 1:
            val_preds_hist = np.concatenate(val_preds)
            val_gts_hist = np.concatenate(val_gts)
    return {
        "train_loss": tr_loss_hist,
        "val_loss": val_loss_hist,
        "train_acc": tr_acc_hist,
        "val_acc": val_acc_hist,
        "epochs": epochs_hist,
        "val_preds": val_preds_hist,
        "val_gts": val_gts_hist,
    }


# Prepare dataset, train/val split and dataloaders (do only ONCE)
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

# Activation function search space
activation_candidates = ["relu", "leakyrelu", "elu", "gelu"]

for act_fn in activation_candidates:
    print(f"\n=== Training with Vision Activation: {act_fn} ===")
    # New model per activation
    model = ClaimVerifier(activation_fn_name=act_fn).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    # Train + evaluate
    result = train_eval_loop(
        model, loaders, optimizer, criterion, num_epochs=10, activation_key=act_fn
    )
    # Store experiment run
    if "mnist_claims" not in experiment_data["activation_fn_tuning"]:
        experiment_data["activation_fn_tuning"]["mnist_claims"] = {}
    experiment_data["activation_fn_tuning"]["mnist_claims"][act_fn] = {
        "metrics": {"train_acc": result["train_acc"], "val_acc": result["val_acc"]},
        "losses": {
            "train": result["train_loss"],
            "val": result["val_loss"],
        },
        "predictions": result["val_preds"],
        "ground_truth": result["val_gts"],
        "epochs": result["epochs"],
    }
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(result["epochs"], result["train_acc"], label="Train Accuracy")
    plt.plot(result["epochs"], result["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Train/Val Accuracy ({act_fn})")
    plt.legend()
    plot_path = os.path.join(working_dir, f"mnist_claims_accuracy_curve_{act_fn}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[{act_fn}] Accuracy curve saved to: {plot_path}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Print summary: best activation
print("\n==== FINAL VALIDATION ACCURACIES ====")
best_acc = -1
best_act = None
for act_fn in activation_candidates:
    val_acc_hist = experiment_data["activation_fn_tuning"]["mnist_claims"][act_fn][
        "metrics"
    ]["val_acc"]
    final_val_acc = val_acc_hist[-1]
    print(f"{act_fn}: final val acc = {final_val_acc:.4f}")
    if final_val_acc > best_acc:
        best_acc = final_val_acc
        best_act = act_fn
print(f"Best Vision Activation Function: {best_act} (val acc {best_acc:.4f})")
