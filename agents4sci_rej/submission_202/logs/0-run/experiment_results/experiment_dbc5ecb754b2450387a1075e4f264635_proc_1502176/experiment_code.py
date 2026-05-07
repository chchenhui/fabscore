import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- Data structure for saving experiment results -----
experiment_data = {
    "augmentation_tuning": {},
}


# ----- Synthetic claim generator -----
def generate_claim(digits):
    claim_type = random.choice(["sum_even", "all_less_than_5"])
    if claim_type == "sum_even":
        label = int(sum(digits) % 2 == 0)
        text = "The sum of the digits is even."
    elif claim_type == "all_less_than_5":
        label = int(all([d < 5 for d in digits]))
        text = "All digits are less than 5."
    return text, label


# ----- MNISTClaimDataset, supports custom transform -----
class MNISTClaimDataset(Dataset):
    def __init__(self, num_samples=3000, tokenizer=None, img_transform=None):
        self.raw_mnist = datasets.MNIST(
            root=".", train=True, download=True, transform=None
        )
        self.num_samples = num_samples
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased")
        self.img_transform = img_transform
        self.samples = self._generate()

    def _generate(self):
        samples = []
        for _ in range(self.num_samples):
            indices = random.sample(range(len(self.raw_mnist)), 3)
            imgs = [self.raw_mnist[i][0] for i in indices]
            labels = [self.raw_mnist[i][1] for i in indices]
            text, truth = generate_claim(labels)
            samples.append((imgs, text, truth))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imgs, text, label = self.samples[idx]
        img_tensors = []
        for img in imgs:
            if self.img_transform:
                img = self.img_transform(img)
            else:
                img = transforms.ToTensor()(img)
            img_tensors.append(img)  # each img: (1,28,28)
        img_tensor = torch.stack(img_tensors)  # (3, 1, 28, 28)
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


# ----- Simple CNN for processing the image -----
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


# ----- Multimodal claim verification model -----
class ClaimVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = CNNVisionEncoder()
        self.text = BertModel.from_pretrained("bert-base-uncased")
        for param in self.text.parameters():
            param.requires_grad = False  # freeze BERT
        self.fc = nn.Sequential(
            nn.Linear(128 + 768, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
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


# ----- Collate function for the dataloader -----
def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])  # (B, 3, 1, 28, 28)
    imgs = imgs.squeeze(2)
    input_ids = torch.stack([item[1] for item in batch])
    attn_mask = torch.stack([item[2] for item in batch])
    labels = torch.stack([item[3] for item in batch])
    return imgs, input_ids, attn_mask, labels


# ----- Training/Evaluation Loop -----
def train_eval_loop(model, loaders, optimizer, criterion, num_epochs=10, exp_log=None):
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

        if exp_log is not None:
            exp_log["metrics"]["train"].append(tr_acc)
            exp_log["metrics"]["val"].append(val_acc)
            exp_log["losses"]["train"].append(tr_loss)
            exp_log["losses"]["val"].append(val_loss)
        if epoch == num_epochs - 1 and exp_log is not None:
            exp_log["predictions"] = np.concatenate(val_preds)
            exp_log["ground_truth"] = np.concatenate(val_gts)
        print(
            f"Epoch {epoch+1}/{num_epochs}: train_acc={tr_acc:.4f}, val_acc={val_acc:.4f}, train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}"
        )
    return model


# ----- Define augmentation grid -----
augmentation_grid = [
    # Each item: (rot_deg, shift_pct, flip_p)
    {"rotation": 0, "translation": 0.0, "flip": 0.0},  # No aug
    {"rotation": 10, "translation": 0.0, "flip": 0.0},
    {"rotation": 0, "translation": 0.1, "flip": 0.0},
    {"rotation": 0, "translation": 0.0, "flip": 0.5},
    {"rotation": 10, "translation": 0.1, "flip": 0.0},
    {"rotation": 10, "translation": 0.0, "flip": 0.5},
    {"rotation": 0, "translation": 0.1, "flip": 0.5},
    {"rotation": 10, "translation": 0.1, "flip": 0.5},
]
augmentation_names = [
    "none",
    "rot10",
    "shift0.1",
    "flip0.5",
    "rot10_shift0.1",
    "rot10_flip0.5",
    "shift0.1_flip0.5",
    "rot10_shift0.1_flip0.5",
]

# ----- Bert Tokenizer (load only once) -----
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ----- Main hyperparam tuning loop -----
for aug_params, aug_name in zip(augmentation_grid, augmentation_names):
    print(f"\n=== Running experiment: {aug_name} | Params: {aug_params} ===")
    # Augmentation pipeline
    tfm_list = []
    if aug_params["flip"] > 0:
        tfm_list.append(transforms.RandomHorizontalFlip(p=aug_params["flip"]))
    if aug_params["rotation"] > 0 and aug_params["translation"] > 0:
        # Use RandomAffine to combine rotation & translation
        tfm_list.append(
            transforms.RandomAffine(
                degrees=aug_params["rotation"],
                translate=(aug_params["translation"], aug_params["translation"]),
            )
        )
    elif aug_params["rotation"] > 0:
        tfm_list.append(transforms.RandomRotation(degrees=aug_params["rotation"]))
    elif aug_params["translation"] > 0:
        tfm_list.append(
            transforms.RandomAffine(
                degrees=0,
                translate=(aug_params["translation"], aug_params["translation"]),
            )
        )
    tfm_list.append(transforms.ToTensor())
    img_transform = transforms.Compose(tfm_list)

    # Dataset
    full_dataset = MNISTClaimDataset(
        num_samples=3000, tokenizer=tokenizer, img_transform=img_transform
    )
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
    # Experiment log
    exp_log = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "aug_params": aug_params,
        "epochs": [],
    }

    # Model (reset each run!)
    model = ClaimVerifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    # Train/Eval
    model = train_eval_loop(
        model, loaders, optimizer, criterion, num_epochs=10, exp_log=exp_log
    )
    exp_log["epochs"] = list(range(1, len(exp_log["metrics"]["train"]) + 1))
    # Store in experiment_data
    experiment_data["augmentation_tuning"][aug_name] = exp_log

    # Save metrics for quick checkpointing each loop
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# ----- Result visualization: Plot val acc for all augmentations -----
plt.figure(figsize=(10, 6))
for aug_name in augmentation_names:
    ep = experiment_data["augmentation_tuning"][aug_name]["epochs"]
    val_acc = experiment_data["augmentation_tuning"][aug_name]["metrics"]["val"]
    plt.plot(ep, val_acc, label=aug_name)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy for Different Augmentation Schemes")
plt.legend()
plt.grid()
plot_path = os.path.join(working_dir, "augmentation_tuning_val_acc_curve.png")
plt.savefig(plot_path)
plt.close()
print(f"All augmentation curves saved to: {plot_path}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Print best result from val acc
best_acc, best_setting = 0.0, None
for aug_name in augmentation_names:
    acc = experiment_data["augmentation_tuning"][aug_name]["metrics"]["val"][-1]
    print(f"Aug: {aug_name:20s} | Final Val Acc: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_setting = aug_name
print(f"Best augmentation: {best_setting} | Validation Acc: {best_acc:.4f}")
