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
from datasets import load_dataset
import matplotlib.pyplot as plt

# --- DATA STRUCTURE FOR EXPERIMENT ---
experiment_data = {
    "claim_diversity_ablation": {
        "mnist": {
            "metrics": {"train": [], "val": [], "train_logic": [], "val_logic": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        },
        "fashion_mnist": {
            "metrics": {"train": [], "val": [], "train_logic": [], "val_logic": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": [],
        },
        "svhn": {
            "metrics": {"train": [], "val": [], "train_logic": [], "val_logic": []},
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


# Ablation: restricts to a single claim type for training
class HF_ClaimDataset_SingleClaim(Dataset):
    def __init__(
        self,
        hfdata,
        num_samples=4000,
        dsname="mnist",
        claim_type="sum_even",
        for_val=False,
    ):
        self.data = hfdata
        self.num_samples = num_samples
        self.tokenizer = tokenizer
        self.dsname = dsname
        if for_val:
            self.available_claims = get_available_claims(dsname)
        else:
            self.available_claims = [claim_type]
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
            torch.tensor(logicvec, dtype=torch.int64),
        )


class HF_SVHNClaimDataset_SingleClaim(Dataset):
    def __init__(self, hfdata, num_samples=4000, claim_type="sum_even", for_val=False):
        self.data = hfdata
        self.num_samples = num_samples
        self.tokenizer = tokenizer
        if for_val:
            self.available_claims = get_available_claims("svhn")
        else:
            self.available_claims = [claim_type]
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
    imgs = imgs.view(len(batch), -1, 28, 28)
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


def logical_consistency_accuracy(preds, gts, logicvecs, claim_labels=None):
    correct = np.round(preds) == gts
    return np.mean(correct)


def train_on_dataset_singleclaim(name, dataset_train, dataset_val, in_c, ablation_key):
    print(f"\nTraining on {name} (claim diversity ablation)...")
    train_len = len(dataset_train)
    val_len = len(dataset_val)
    train_loader = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    model = ClaimVerifier(in_c, do_finetune_text=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct = 0.0, 0
        all_preds, all_gts, all_logicvecs = [], [], []
        for imgs, input_ids, attn_mask, labels, logicvec in train_loader:
            imgs, input_ids, attn_mask, labels = (
                imgs.to(device).float(),
                input_ids.to(device),
                attn_mask.to(device),
                labels.to(device),
            )
            output = model(imgs, input_ids, attn_mask)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            preds = (output > 0.5).float()
            correct += (preds == labels).sum().item()
            all_preds.append(preds.detach().cpu().numpy())
            all_gts.append(labels.detach().cpu().numpy())
            all_logicvecs.append(logicvec.cpu().numpy())
        train_acc = correct / train_len
        train_loss = total_loss / train_len
        tpreds = np.concatenate(all_preds)
        tgts = np.concatenate(all_gts)
        tlogicvecs = np.concatenate(all_logicvecs)
        train_logic = logical_consistency_accuracy(tpreds, tgts, tlogicvecs)
        experiment_data[ablation_key][name]["metrics"]["train"].append(train_acc)
        experiment_data[ablation_key][name]["losses"]["train"].append(train_loss)
        experiment_data[ablation_key][name]["metrics"]["train_logic"].append(
            train_logic
        )
        # Validation
        model.eval()
        val_loss, vcorrect = 0.0, 0
        all_preds, all_gts, all_logicvecs = [], [], []
        with torch.no_grad():
            for imgs, input_ids, attn_mask, labels, logicvec in val_loader:
                imgs, input_ids, attn_mask, labels = (
                    imgs.to(device).float(),
                    input_ids.to(device),
                    attn_mask.to(device),
                    labels.to(device),
                )
                output = model(imgs, input_ids, attn_mask)
                loss = criterion(output, labels)
                val_loss += loss.item() * len(labels)
                preds = (output > 0.5).float()
                vcorrect += (preds == labels).sum().item()
                all_preds.append(preds.detach().cpu().numpy())
                all_gts.append(labels.detach().cpu().numpy())
                all_logicvecs.append(logicvec.cpu().numpy())
        val_acc = vcorrect / val_len
        val_loss = val_loss / val_len
        vpreds = np.concatenate(all_preds)
        vgts = np.concatenate(all_gts)
        vlogicvecs = np.concatenate(all_logicvecs)
        val_logic = logical_consistency_accuracy(vpreds, vgts, vlogicvecs)
        experiment_data[ablation_key][name]["metrics"]["val"].append(val_acc)
        experiment_data[ablation_key][name]["losses"]["val"].append(val_loss)
        experiment_data[ablation_key][name]["metrics"]["val_logic"].append(val_logic)
        experiment_data[ablation_key][name]["epochs"].append(epoch + 1)
        print(
            f"Epoch {epoch+1}: validation_loss = {val_loss:.4f}  | val_acc = {val_acc:.4f}  | val_logic_acc = {val_logic:.4f}"
        )
        if epoch == NUM_EPOCHS - 1:
            experiment_data[ablation_key][name]["predictions"] = vpreds
            experiment_data[ablation_key][name]["ground_truth"] = vgts


def plot_metric_curve_ablation(ablation_key, metrickey, ylabel, fname):
    plt.figure(figsize=(8, 6))
    colors = ["b", "r", "g"]
    for i, dsname in enumerate(["mnist", "fashion_mnist", "svhn"]):
        plt.plot(
            experiment_data[ablation_key][dsname]["epochs"],
            experiment_data[ablation_key][dsname]["metrics"][metrickey],
            color=colors[i],
            label=dsname,
        )
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} (Claim Diversity Ablation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()


# --- MAIN EXECUTION (ablation) ---
claim_type = "sum_even"  # You can change to another single claim to experiment

mnist_hf = load_dataset("mnist", split="train")
mnist_train = HF_ClaimDataset_SingleClaim(
    mnist_hf, num_samples=2720, dsname="mnist", claim_type=claim_type, for_val=False
)
mnist_val = HF_ClaimDataset_SingleClaim(
    mnist_hf, num_samples=480, dsname="mnist", claim_type=claim_type, for_val=True
)
train_on_dataset_singleclaim(
    "mnist", mnist_train, mnist_val, in_c=9, ablation_key="claim_diversity_ablation"
)

fmnist_hf = load_dataset("fashion_mnist", split="train")
fmnist_train = HF_ClaimDataset_SingleClaim(
    fmnist_hf,
    num_samples=2720,
    dsname="fashion_mnist",
    claim_type=claim_type,
    for_val=False,
)
fmnist_val = HF_ClaimDataset_SingleClaim(
    fmnist_hf,
    num_samples=480,
    dsname="fashion_mnist",
    claim_type=claim_type,
    for_val=True,
)
train_on_dataset_singleclaim(
    "fashion_mnist",
    fmnist_train,
    fmnist_val,
    in_c=9,
    ablation_key="claim_diversity_ablation",
)

svhn_hf = load_dataset("svhn", "cropped_digits", split="train")
svhn_train = HF_SVHNClaimDataset_SingleClaim(
    svhn_hf, num_samples=2720, claim_type=claim_type, for_val=False
)
svhn_val = HF_SVHNClaimDataset_SingleClaim(
    svhn_hf, num_samples=480, claim_type=claim_type, for_val=True
)
train_on_dataset_singleclaim(
    "svhn", svhn_train, svhn_val, in_c=9, ablation_key="claim_diversity_ablation"
)

plot_metric_curve_ablation(
    "claim_diversity_ablation",
    "val",
    "Validation Accuracy",
    "val_acc_compare_ablation.png",
)
plot_metric_curve_ablation(
    "claim_diversity_ablation",
    "val_logic",
    "Logical Consistency Accuracy",
    "val_logic_acc_compare_ablation.png",
)

# Overlay for each dataset
for dsname in ["mnist", "fashion_mnist", "svhn"]:
    plt.figure()
    plt.plot(
        experiment_data["claim_diversity_ablation"][dsname]["epochs"],
        experiment_data["claim_diversity_ablation"][dsname]["metrics"]["val"],
        label="Val Acc",
    )
    plt.plot(
        experiment_data["claim_diversity_ablation"][dsname]["epochs"],
        experiment_data["claim_diversity_ablation"][dsname]["metrics"]["val_logic"],
        label="Logic Acc",
    )
    plt.xlabel("Epoch")
    plt.legend()
    plt.title(f"{dsname} (Ablation) - Accuracies per Epoch")
    plt.savefig(os.path.join(working_dir, f"{dsname}_acc_ablation.png"))
    plt.close()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

for dsname in ["mnist", "fashion_mnist", "svhn"]:
    logic = experiment_data["claim_diversity_ablation"][dsname]["metrics"]["val_logic"][
        -1
    ]
    print(f"Final Logical Consistency Accuracy ({dsname}, ablation): {logic:.4f}")
