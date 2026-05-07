import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from datasets import load_dataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

experiment_data = {
    "mnist": {
        "metrics": {"train_acc": [], "val_acc": [], "train_logic": [], "val_logic": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    },
    "fashion_mnist": {
        "metrics": {"train_acc": [], "val_acc": [], "train_logic": [], "val_logic": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    },
    "svhn": {
        "metrics": {"train_acc": [], "val_acc": [], "train_logic": [], "val_logic": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    },
}

BATCH_SIZE = 64
LR = 1e-4
NUM_EPOCHS = 10
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def pad_image(img, target_size=(28, 28)):
    """
    Pads or crops a tensor or np.array (C,H,W) to (C,target_h,target_w).
    Pads with zeros if image is smaller than target, crops center if larger.
    """
    if isinstance(img, np.ndarray):
        # (C, H, W)
        C = img.shape[0]
        h, w = img.shape[1:3]
        th, tw = target_size
        if (h, w) == (th, tw):
            return img
        res = np.zeros((C, th, tw), dtype=img.dtype)
        # Decide cropping if needed
        h_start_img, w_start_img = 0, 0
        h_end_img, w_end_img = h, w
        h_start_res, w_start_res = 0, 0
        # Cropping
        if h > th:
            h_start_img = (h - th) // 2
            h_end_img = h_start_img + th
            h_start_res = 0
        else:
            h_end_img = h
            h_start_res = (th - h) // 2
        if w > tw:
            w_start_img = (w - tw) // 2
            w_end_img = w_start_img + tw
            w_start_res = 0
        else:
            w_end_img = w
            w_start_res = (tw - w) // 2

        # Fill res only in overlap
        h_copy = min(h, th)
        w_copy = min(w, tw)
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
        h_start_img, w_start_img = 0, 0
        h_end_img, w_end_img = h, w
        h_start_res, w_start_res = 0, 0
        # Cropping
        if h > th:
            h_start_img = (h - th) // 2
            h_end_img = h_start_img + th
            h_start_res = 0
        else:
            h_end_img = h
            h_start_res = (th - h) // 2
        if w > tw:
            w_start_img = (w - tw) // 2
            w_end_img = w_start_img + tw
            w_start_res = 0
        else:
            w_end_img = w
            w_start_res = (tw - w) // 2
        # Fill res only in overlap
        h_copy = min(h, th)
        w_copy = min(w, tw)
        res[
            :, h_start_res : h_start_res + h_copy, w_start_res : w_start_res + w_copy
        ] = img[
            :, h_start_img : h_start_img + h_copy, w_start_img : w_start_img + w_copy
        ]
        return res
    else:
        raise TypeError("Unknown input type")


class HF_MNISTClaimDataset(Dataset):
    # Allows use with both MNIST and Fashion-MNIST (HF)
    def __init__(
        self,
        hfdata,
        num_samples=2000,
        claim_tasks=("sum_even", "all_lt_5"),
        tokenizer=None,
    ):
        self.data = hfdata
        self.num_samples = num_samples
        # For MNIST and FashionMNIST: classes == 10
        self.classes = int(max(self.data["label"])) + 1
        self.claim_tasks = claim_tasks
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased")
        self.samples = self._generate()

    def _generate(self):
        samples = []
        for _ in range(self.num_samples):
            indices = random.sample(range(len(self.data)), 3)
            imgs = []
            labels = []
            for i in indices:
                arr = np.array(self.data[i]["image"]).astype(np.float32) / 255.0
                arr3 = np.repeat(arr[None, :, :], 3, axis=0)
                arr3 = pad_image(arr3, target_size=(28, 28))
                imgs.append(torch.from_numpy(arr3))
                labels.append(self.data[i]["label"])
            claim_type = random.choice(self.claim_tasks)
            if claim_type == "sum_even":
                truelab = int((sum(labels) % 2) == 0)
                text = (
                    "The sum of the digits is even."
                    if self.classes == 10
                    else "The sum of the classes is even."
                )
            elif claim_type == "all_lt_5":
                truelab = int(all([l < 5 for l in labels]))
                text = (
                    "All digits are less than 5."
                    if self.classes == 10
                    else "All items have label less than 5."
                )
            else:
                raise ValueError("Unknown claim")
            samples.append((torch.stack(imgs, 0), text, truelab, labels))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imgs, text, label, logic_subparts = self.samples[idx]
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
            torch.tensor(logic_subparts, dtype=torch.long),
        )


class HF_SVHNClaimDataset(Dataset):
    def __init__(self, hfdata, num_samples=2000, tokenizer=None):
        self.data = hfdata
        self.num_samples = num_samples
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased")
        self.samples = self._generate()

    def _generate(self):
        samples = []
        for _ in range(self.num_samples):
            indices = random.sample(range(len(self.data)), 3)
            imgs = []
            labels = []
            for i in indices:
                arr = np.array(self.data[i]["image"], dtype=np.float32) / 255.0
                # HF SVHN images are (H, W, C); convert to (C, H, W)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    arr = np.transpose(arr, (2, 0, 1))  # Make (C, H, W)
                arr28 = pad_image(arr, target_size=(28, 28))  # ensure (3,28,28)
                imgs.append(torch.from_numpy(arr28))
                labels.append(self.data[i]["label"])
            claim_type = random.choice(["sum_odd", "all_d_gt_2"])
            if claim_type == "sum_odd":
                truelab = int((sum(labels) % 2) == 1)
                text = "The sum of house numbers is odd."
            elif claim_type == "all_d_gt_2":
                truelab = int(all([l > 2 for l in labels]))
                text = "All house digits are greater than 2."
            else:
                raise ValueError()
            samples.append((torch.stack(imgs, 0), text, truelab, labels))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imgs, text, label, logic_subparts = self.samples[idx]
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
            torch.tensor(logic_subparts, dtype=torch.long),
        )


class CNNVisionEncoder(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
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


class ClaimVerifier(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.vision = CNNVisionEncoder(input_channels=in_c)
        self.text = BertModel.from_pretrained("bert-base-uncased")
        for param in self.text.parameters():
            param.requires_grad = False  # Freeze text encoder for efficiency
        self.fc = nn.Sequential(
            nn.Linear(128 + 768, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, imgs, input_ids, attn_mask):
        vis_feat = self.vision(imgs)
        txt_feat = self.text(
            input_ids=input_ids, attention_mask=attn_mask
        ).last_hidden_state[:, 0, :]
        combined = torch.cat([vis_feat, txt_feat], dim=1)
        return self.fc(combined).squeeze(1)


def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])  # shape (B, 3, 3,28,28)
    if imgs.dim() == 5 and imgs.shape[2:] == (3, 28, 28):
        imgs = imgs.view(-1, 3, 28, 28)
        input_ids = torch.stack([item[1] for item in batch])
        attn_mask = torch.stack([item[2] for item in batch])
        input_ids = input_ids.repeat_interleave(3, dim=0)
        attn_mask = attn_mask.repeat_interleave(3, dim=0)
        labels = torch.tensor([item[3] for item in batch], dtype=torch.float32)
        labels = labels.repeat_interleave(3, dim=0)
        logic_subparts = torch.stack([item[4] for item in batch])
        logic_subparts = logic_subparts.view(-1, 3)
    else:
        input_ids = torch.stack([item[1] for item in batch])
        attn_mask = torch.stack([item[2] for item in batch])
        labels = torch.tensor([item[3] for item in batch], dtype=torch.float32)
        logic_subparts = torch.stack([item[4] for item in batch])
    return imgs, input_ids, attn_mask, labels, logic_subparts


def logical_consistency_accuracy(preds, gts, logic_parts, claim_type):
    preds = np.round(preds)
    logic_acc = []
    for p, gt, lvec in zip(preds, gts, logic_parts):
        if p == gt:
            logic_acc.append(1)
        else:
            logic_acc.append(0)
    return np.mean(logic_acc)


def run_experiment_on_dataset(name, dataset, in_c):
    print(f"\nTraining on {name} ...")
    train_len = int(0.8 * len(dataset))
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
    loaders = {"train": train_loader, "val": val_loader}

    model = ClaimVerifier(in_c).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.BCELoss()

    accs, logic_accs, losses, val_accs, val_logic_accs, val_losses, epochs = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct, n, logic_c, logic_vecs = 0, 0, 0, 0, []
        for imgs, input_ids, attn_mask, labels, logic_subparts in loaders["train"]:
            imgs = imgs.to(device).float()
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            outputs = model(imgs, input_ids, attn_mask)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            logic_vecs.append(logic_subparts.cpu().numpy())
            n += imgs.size(0)
        train_acc = correct / n
        losses.append(total_loss / n)
        accs.append(train_acc)

        # Compute train logic acc
        logic_preds, logic_gts = (
            preds.detach().cpu().numpy(),
            labels.detach().cpu().numpy(),
        )
        all_logic_subparts = np.concatenate(logic_vecs, axis=0)
        train_logic_acc = logical_consistency_accuracy(
            logic_preds, logic_gts, all_logic_subparts, None
        )
        logic_accs.append(train_logic_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_n, vlogic_vecs, val_preds, val_gts = (
            0,
            0,
            0,
            [],
            [],
            [],
        )
        with torch.no_grad():
            for imgs, input_ids, attn_mask, labels, logic_subparts in loaders["val"]:
                imgs = imgs.to(device).float()
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device)
                outputs = model(imgs, input_ids, attn_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = (outputs > 0.5).float().cpu().numpy()
                val_preds.append(preds)
                val_gts.append(labels.cpu().numpy())
                vlogic_vecs.append(logic_subparts.cpu().numpy())
                val_correct += (preds == labels.cpu().numpy()).sum()
                val_n += imgs.size(0)
        vpreds, vgts = np.concatenate(val_preds), np.concatenate(val_gts)
        vlogic_all = np.concatenate(vlogic_vecs, axis=0)
        val_logic_acc = logical_consistency_accuracy(vpreds, vgts, vlogic_all, None)
        val_acc = val_correct / val_n
        val_losses.append(val_loss / val_n)
        val_accs.append(val_acc)
        val_logic_accs.append(val_logic_acc)
        epochs.append(epoch + 1)

        print(
            f"Epoch {epoch+1}: val_loss = {val_losses[-1]:.4f}, val_acc = {val_acc:.4f}, val_logic_acc = {val_logic_acc:.4f}"
        )

    experiment_data[name]["metrics"]["train_acc"] = accs
    experiment_data[name]["metrics"]["val_acc"] = val_accs
    experiment_data[name]["metrics"]["train_logic"] = logic_accs
    experiment_data[name]["metrics"]["val_logic"] = val_logic_accs
    experiment_data[name]["losses"]["train"] = losses
    experiment_data[name]["losses"]["val"] = val_losses
    experiment_data[name]["epochs"] = epochs
    experiment_data[name]["predictions"] = vpreds
    experiment_data[name]["ground_truth"] = vgts


def plot_metric_curve(metrickey, ylabel, fname):
    plt.figure(figsize=(8, 6))
    colors = ["b", "r", "g"]
    for i, dsname in enumerate(["mnist", "fashion_mnist", "svhn"]):
        plt.plot(
            experiment_data[dsname]["epochs"],
            experiment_data[dsname]["metrics"][metrickey],
            color=colors[i],
            label=dsname,
        )
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} on Multimodal Scientific Claim Tasks")
    plt.legend()
    savepath = os.path.join(working_dir, fname)
    plt.savefig(savepath)
    plt.close()


# --- Run the experiments ---
mnist_hf = load_dataset("mnist", split="train")
mnist_claim_set = HF_MNISTClaimDataset(
    mnist_hf,
    num_samples=1500,
    claim_tasks=("sum_even", "all_lt_5"),
    tokenizer=tokenizer,
)
run_experiment_on_dataset("mnist", mnist_claim_set, in_c=3)

fmnist_hf = load_dataset("fashion_mnist", split="train")
fmnist_claim_set = HF_MNISTClaimDataset(
    fmnist_hf,
    num_samples=1500,
    claim_tasks=("sum_even", "all_lt_5"),
    tokenizer=tokenizer,
)
run_experiment_on_dataset("fashion_mnist", fmnist_claim_set, in_c=3)

svhn_hf = load_dataset("svhn", "cropped_digits", split="train")
svhn_claim_set = HF_SVHNClaimDataset(svhn_hf, num_samples=1500, tokenizer=tokenizer)
run_experiment_on_dataset("svhn", svhn_claim_set, in_c=3)

plot_metric_curve("val_acc", "Validation Accuracy", "val_acc_compare.png")
plot_metric_curve(
    "val_logic", "Logical Consistency Accuracy", "val_logic_acc_compare.png"
)
for dsname in ["mnist", "fashion_mnist", "svhn"]:
    plt.figure()
    plt.plot(
        experiment_data[dsname]["epochs"],
        experiment_data[dsname]["metrics"]["val_acc"],
        label="Val Acc",
    )
    plt.plot(
        experiment_data[dsname]["epochs"],
        experiment_data[dsname]["metrics"]["val_logic"],
        label="Logical Consistency Acc",
    )
    plt.xlabel("Epoch")
    plt.legend()
    plt.title(f"{dsname} - Accuracies Across Epochs")
    plt.savefig(os.path.join(working_dir, f"{dsname}_acc.png"))
    plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

for dsname in ["mnist", "fashion_mnist", "svhn"]:
    logic = experiment_data[dsname]["metrics"]["val_logic"][-1]
    print(f"Final Logical Consistency Accuracy ({dsname}): {logic:.4f}")
