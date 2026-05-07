import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import matplotlib.pyplot as plt

# Set experiment data structure for ablation study
experiment_data = {
    "no_logic_supervision": {
        "mnist": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        },
        "fashion_mnist": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        },
        "svhn": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        },
    }
}

BATCH_SIZE = 64
LR = 1e-4
NUM_EPOCHS = 50
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def pad_image(img, target_size=(28, 28)):
    if isinstance(img, np.ndarray):
        C = img.shape[0]
        h, w = img.shape[1:3]
        th, tw = target_size
        if (h, w) == (th, tw):
            return img
        res = np.zeros((C, th, tw), dtype=img.dtype)
        h_copy = min(h, th)
        w_copy = min(w, tw)
        h_start_res = (th - h) // 2 if h < th else 0
        w_start_res = (tw - w) // 2 if w < tw else 0
        h_start_img = (h - th) // 2 if h > th else 0
        w_start_img = (w - tw) // 2 if w > tw else 0
        res[
            :, h_start_res : h_start_res + h_copy, w_start_res : w_start_res + w_copy
        ] = img[
            :, h_start_img : h_start_img + h_copy, w_start_img : w_start_img + w_copy
        ]
        return res
    elif isinstance(img, torch.Tensor):
        C = img.shape[0]
        h, w = img.shape[1:3]
        th, tw = target_size
        if (h, w) == (th, tw):
            return img
        res = torch.zeros(C, th, tw, dtype=img.dtype)
        h_copy = min(h, th)
        w_copy = min(w, tw)
        h_start_res = (th - h) // 2 if h < th else 0
        w_start_res = (tw - w) // 2 if w < tw else 0
        h_start_img = (h - th) // 2 if h > th else 0
        w_start_img = (w - tw) // 2 if w > tw else 0
        res[
            :, h_start_res : h_start_res + h_copy, w_start_res : w_start_res + w_copy
        ] = img[
            :, h_start_img : h_start_img + h_copy, w_start_img : w_start_img + w_copy
        ]
        return res
    else:
        raise TypeError("Unknown type for pad_image")


def generate_claim(labels, available_claims):
    ctype = random.choice(available_claims)
    L = labels
    if ctype == "sum_even":
        label = int((sum(L) % 2) == 0)
        text = "The sum of the digits is even."
        subparts = [l for l in L]
    elif ctype == "all_lt_5":
        label = int(all(l < 5 for l in L))
        text = "All digits are less than 5."
        subparts = [l < 5 for l in L]
    elif ctype == "exactly_two_odd":
        label = int(sum([l % 2 == 1 for l in L]) == 2)
        text = "Exactly two digits are odd numbers."
        subparts = [l % 2 == 1 for l in L]
    elif ctype == "at_least_one_is_7":
        label = int(any(l == 7 for l in L))
        text = "At least one digit is seven."
        subparts = [l == 7 for l in L]
    elif ctype == "all_unique":
        label = int(len(set(L)) == len(L))
        text = "All three digits are unique."
        uniqarr = [L.count(l) == 1 for l in L]
        subparts = uniqarr
    else:
        raise ValueError()
    return text, label, np.array(subparts, dtype=np.int64)


def get_available_claims(dsname):
    return [
        "sum_even",
        "all_lt_5",
        "exactly_two_odd",
        "at_least_one_is_7",
        "all_unique",
    ]


class HF_ClaimDataset(Dataset):
    def __init__(self, hfdata, num_samples=4000, dsname="mnist"):
        self.data = hfdata
        self.num_samples = num_samples
        self.tokenizer = tokenizer
        self.dsname = dsname
        self.available_claims = get_available_claims(dsname)
        self.samples = self._generate()

    def _generate(self):
        N = len(self.data)
        samples = []
        for _ in range(self.num_samples):
            indices = random.sample(range(N), 3)
            imgs = []
            labels = []
            for i in indices:
                arr = np.array(self.data[i]["image"]).astype(np.float32) / 255.0
                arr3 = np.repeat(arr[None, :, :], 3, axis=0)
                arr3 = pad_image(arr3, target_size=(28, 28))
                imgs.append(torch.from_numpy(arr3))
                labels.append(self.data[i]["label"])
            text, truth, logicvec = generate_claim(labels, self.available_claims)
            samples.append(
                (torch.stack(imgs, 0), text, truth, np.array(labels), logicvec)
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imgs, text, label, digits, logicvec = self.samples[idx]
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)
        return (
            imgs,
            input_ids,
            attn_mask,
            torch.tensor(label, dtype=torch.float32),
            # logicvec still returned, but not used
            torch.tensor(logicvec, dtype=torch.int64),
        )


class HF_SVHNClaimDataset(Dataset):
    def __init__(self, hfdata, num_samples=4000):
        self.data = hfdata
        self.num_samples = num_samples
        self.tokenizer = tokenizer
        self.available_claims = get_available_claims("svhn")
        self.samples = self._generate()

    def _generate(self):
        N = len(self.data)
        samples = []
        for _ in range(self.num_samples):
            indices = random.sample(range(N), 3)
            imgs = []
            labels = []
            for i in indices:
                arr = np.array(self.data[i]["image"], dtype=np.float32) / 255.0
                if arr.ndim == 3 and arr.shape[2] == 3:
                    arr = np.transpose(arr, (2, 0, 1))
                arr28 = pad_image(arr, target_size=(28, 28))
                imgs.append(torch.from_numpy(arr28))
                labels.append(self.data[i]["label"])
            text, truth, logicvec = generate_claim(labels, self.available_claims)
            samples.append(
                (torch.stack(imgs, 0), text, truth, np.array(labels), logicvec)
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imgs, text, label, digits, logicvec = self.samples[idx]
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)
        return (
            imgs,
            input_ids,
            attn_mask,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(logicvec, dtype=torch.int64),
        )


def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    imgs = imgs.view(len(batch), -1, 28, 28)  # (B,9,28,28)
    input_ids = torch.stack([item[1] for item in batch])
    attn_mask = torch.stack([item[2] for item in batch])
    labels = torch.stack([item[3] for item in batch])
    logicvec = torch.stack([item[4] for item in batch])
    return imgs, input_ids, attn_mask, labels, logicvec


class CNNVisionEncoder(nn.Module):
    def __init__(self, input_channels=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ClaimVerifier(nn.Module):
    def __init__(self, in_c, do_finetune_text=True):
        super().__init__()
        self.vision = CNNVisionEncoder(input_channels=in_c)
        self.text = BertModel.from_pretrained("bert-base-uncased")
        for param in self.text.parameters():
            param.requires_grad = do_finetune_text
        self.fc = nn.Sequential(
            nn.Linear(256 + 768, 192),
            nn.ReLU(),
            nn.Linear(192, 1),
            nn.Sigmoid(),
        )

    def forward(self, imgs, input_ids, attn_mask):
        vis_feat = self.vision(imgs)
        txt_feat = self.text(
            input_ids=input_ids, attention_mask=attn_mask
        ).last_hidden_state[:, 0, :]
        x = torch.cat([vis_feat, txt_feat], dim=1)
        return self.fc(x).squeeze(1)


def train_on_dataset_no_logic(name, dataset, in_c):
    ablation_key = "no_logic_supervision"
    print(f"\n[NO LOGIC SUPERVISION] Training on {name} ...")
    train_len = int(0.85 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(
        dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    model = ClaimVerifier(in_c, do_finetune_text=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    expdat = experiment_data[ablation_key][name]

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        all_preds, all_labels = [], []
        for imgs, input_ids, attn_mask, labels, _logicvec in train_loader:
            imgs = imgs.to(device).float()
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            output = model(imgs, input_ids, attn_mask)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            preds = (output > 0.5).float()
            correct += (preds == labels).sum().item()
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
        ntrain = train_len
        train_acc = correct / ntrain
        train_loss = total_loss / ntrain
        expdat["metrics"]["train"].append(train_acc)
        expdat["losses"]["train"].append(train_loss)
        # Validation
        model.eval()
        val_loss = 0.0
        vcorrect = 0
        vall_preds, vall_labels = [], []
        with torch.no_grad():
            for imgs, input_ids, attn_mask, labels, _logicvec in val_loader:
                imgs = imgs.to(device).float()
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device)
                output = model(imgs, input_ids, attn_mask)
                loss = criterion(output, labels)
                val_loss += loss.item() * len(labels)
                preds = (output > 0.5).float()
                vcorrect += (preds == labels).sum().item()
                vall_preds.append(preds.detach().cpu().numpy())
                vall_labels.append(labels.detach().cpu().numpy())
        nval = val_len
        val_acc = vcorrect / nval
        val_loss = val_loss / nval
        expdat["metrics"]["val"].append(val_acc)
        expdat["losses"]["val"].append(val_loss)
        expdat["epochs"].append(epoch + 1)
        print(
            f"[No logic] Epoch {epoch+1}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )
        # Store validation predictions on last epoch
        if epoch == NUM_EPOCHS - 1:
            expdat["predictions"] = np.concatenate(vall_preds)
            expdat["ground_truth"] = np.concatenate(vall_labels)


def plot_metric_curve_no_logic(metrickey, ylabel, fname):
    plt.figure(figsize=(8, 6))
    colors = ["b", "r", "g"]
    ablation_key = "no_logic_supervision"
    datasets = ["mnist", "fashion_mnist", "svhn"]
    for i, dsname in enumerate(datasets):
        plt.plot(
            experiment_data[ablation_key][dsname]["epochs"],
            experiment_data[ablation_key][dsname]["metrics"][metrickey],
            color=colors[i],
            label=dsname,
        )
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} across datasets (No Logic Supervision)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()


# Prepare and run ablation on all datasets
mnist_hf = load_dataset("mnist", split="train")
mnist_ds = HF_ClaimDataset(mnist_hf, num_samples=3200, dsname="mnist")
train_on_dataset_no_logic("mnist", mnist_ds, in_c=9)

fmnist_hf = load_dataset("fashion_mnist", split="train")
fmnist_ds = HF_ClaimDataset(fmnist_hf, num_samples=3200, dsname="fashion_mnist")
train_on_dataset_no_logic("fashion_mnist", fmnist_ds, in_c=9)

svhn_hf = load_dataset("svhn", "cropped_digits", split="train")
svhn_ds = HF_SVHNClaimDataset(svhn_hf, num_samples=3200)
train_on_dataset_no_logic("svhn", svhn_ds, in_c=9)

plot_metric_curve_no_logic("val", "Validation Accuracy", "no_logic_val_acc_compare.png")

for dsname in ["mnist", "fashion_mnist", "svhn"]:
    plt.figure()
    plt.plot(
        experiment_data["no_logic_supervision"][dsname]["epochs"],
        experiment_data["no_logic_supervision"][dsname]["metrics"]["val"],
        label="Val Acc",
    )
    plt.xlabel("Epoch")
    plt.legend()
    plt.title(f"{dsname} - Validation Acc per Epoch (No Logic Supervision)")
    plt.savefig(os.path.join(working_dir, f"{dsname}_no_logic_acc.png"))
    plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

for dsname in ["mnist", "fashion_mnist", "svhn"]:
    val_acc = experiment_data["no_logic_supervision"][dsname]["metrics"]["val"][-1]
    print(f"[No Logic] Final Validation Accuracy ({dsname}): {val_acc:.4f}")
