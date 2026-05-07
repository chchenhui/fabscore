import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import scipy.io
from PIL import Image
import timm
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json # Import json to read results.json
import torch.nn.functional as F

# --- Re-define necessary classes from run_experiments.py ---
# (Assuming these are the exact definitions from run_experiments.py)

def generate_heatmaps(keypoints, output_res, sigma=2):
    heatmaps = np.zeros((keypoints.shape[0], keypoints.shape[1], output_res[0], output_res[1]), dtype=np.float32)
    for i in range(keypoints.shape[0]):
        for j in range(keypoints.shape[1]):
            x, y = keypoints[i, j]
            if x >= 0 and y >= 0 and x < output_res[1] and y < output_res[0]:
                ul = [int(x - sigma), int(y - sigma)]
                br = [int(x + sigma + 1), int(y + sigma + 1)]
                size = 2 * sigma + 1
                x_gauss = np.arange(0, size, 1, np.float32)
                y_gauss = x_gauss[:, np.newaxis]
                x0 = y0 = size // 2
                g = np.exp(- ((x_gauss - x0) ** 2 + (y_gauss - y0) ** 2) / (2 * sigma ** 2))
                g_x = max(0, -ul[0]), min(br[0], output_res[1]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], output_res[0]) - ul[1]
                img_x = max(0, ul[0]), min(br[0], output_res[1])
                img_y = max(0, ul[1]), min(br[1], output_res[0])
                heatmaps[i, j, img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return torch.from_numpy(heatmaps)

class ResearchDataset(Dataset):
    """Custom Dataset for loading the Data."""
    def __init__(self, data_path, transform=None, mode='train'):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        self.image_paths = []
        self.labels = []

        class_map = {"Normal": 0, "Large": 1, "Small": 2}

        for class_name, class_idx in class_map.items():
            class_path = os.path.join(self.data_path, self.mode, 'Images_classes', class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    base_filename, _ = os.path.splitext(img_name)
                    self.image_paths.append(os.path.join(self.data_path, self.mode, 'Images', img_name))
                    self.labels.append({
                        'class': class_idx,
                        'keypoints': os.path.join(self.data_path, self.mode, 'Labels', base_filename + '.mat')
                    })

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        mat_path = self.labels[idx]['keypoints']
        mat = scipy.io.loadmat(mat_path)
        keypoints = torch.from_numpy(mat['six_points']).float()
        vhs = torch.from_numpy(mat['VHS']).float().squeeze()

        class_label = self.labels[idx]['class']

        if self.transform:
            image = self.transform(image)

        return image, keypoints, class_label, vhs


class Preprocessor:
    """Preprocessing pipeline."""
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_transforms(self):
        return self.train_transform, self.val_test_transform


class ProposedModel(nn.Module):
    """The proposed three-task model."""
    def __init__(self, num_classes=3, num_keypoints=6, model_size='small', backbone_type='vit', use_cross_attention=True, kp_head_type='hrnet'):
        super(ProposedModel, self).__init__()
        self.backbone_type = backbone_type
        self.use_cross_attention = use_cross_attention

        if backbone_type == 'vit':
            if model_size == 'small':
                self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True)
                embed_dim = 384
            else:
                self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
                embed_dim = 768
            self.backbone.head = nn.Identity()  # Remove the original classification head
        elif backbone_type == 'resnet':
            self.backbone = timm.create_model('resnet50', pretrained=True)
            embed_dim = self.backbone.fc.in_features # This is 2048 for ResNet50
            self.backbone.fc = nn.Identity() # Remove the original classification head
        else:
            raise ValueError("Unsupported backbone type")

        # Keypoint detection head
        if kp_head_type == 'hrnet':
            self.keypoint_head = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, num_keypoints, kernel_size=1)
            )
        elif kp_head_type == 'simple':
            self.keypoint_head = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, num_keypoints, kernel_size=1)
            )
        else:
            raise ValueError("Unsupported keypoint head type")

        # Classification head
        self.classification_head = nn.Linear(embed_dim, num_classes)

        # VHS regression head
        self.vhs_head = nn.Linear(embed_dim, 1)

        # Cross-attention mechanism
        if self.use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8)

    def forward(self, x):
        if self.backbone_type == 'vit':
            features = self.backbone.forward_features(x) # (batch_size, seq_len, embed_dim)
            cls_token = features[:, 0] # (batch_size, embed_dim)
            # patch_tokens for ViT needs to be reshaped to 4D for ConvTranspose2d
            # The reshape dimensions depend on the ViT patch size and image size
            # For vit_small_patch16_224, it's 14x14 patches, embed_dim 384
            patch_tokens = features[:, 1:].permute(0, 2, 1).reshape(x.shape[0], self.backbone.embed_dim, 14, 14) # Corrected reshape
        elif self.backbone_type == 'resnet':
            features = self.backbone(x) # (batch_size, channels, H, W)
            cls_token = features.mean([-2, -1]) # Global average pooling for ResNet (batch_size, channels)
            patch_tokens = features # Use feature map directly for ResNet (batch_size, channels, H, W)

        keypoint_heatmaps = self.keypoint_head(patch_tokens)

        if self.use_cross_attention:
            # For cross-attention, query (cls_token) and key/value (features) need to be (seq_len, batch_size, embed_dim)
            # cls_token is (batch_size, embed_dim) -> (1, batch_size, embed_dim)
            # features for ViT is (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
            # features for ResNet is (batch_size, channels, H, W) -> (batch_size, H*W, channels) -> (H*W, batch_size, channels)
            
            query = cls_token.unsqueeze(0) # (1, batch_size, embed_dim)
            
            if self.backbone_type == 'vit':
                key_value = features.permute(1, 0, 2) # (seq_len, batch_size, embed_dim)
            elif self.backbone_type == 'resnet':
                # Flatten spatial dimensions and permute for MultiheadAttention
                key_value = features.flatten(2).permute(2, 0, 1) # (H*W, batch_size, embed_dim)
            
            attn_output, _ = self.cross_attention(query, key_value, key_value)
            attn_output = attn_output.squeeze(0) # (batch_size, embed_dim)
        else:
            attn_output = cls_token

        # Classification and VHS regression
        class_logits = self.classification_head(attn_output)
        vhs_pred = self.vhs_head(attn_output)

        return keypoint_heatmaps, class_logits, vhs_pred

# --- End of re-defined classes ---


def get_predictions(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, _, class_labels, _ in data_loader:
            images = images.to(device)
            _, cls_logits, _ = model(images)
            preds = torch.argmax(cls_logits, dim=1)
            all_labels.extend(class_labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_combined_confusion_matrices(val_cm, test_cm, class_names, val_accuracy, test_accuracy, filename):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7)) # Two subplots, wider figure

    # Plot Validation Confusion Matrix
    sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'Validation Confusion Matrix (Accuracy: {val_accuracy:.1%})')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Plot Test Confusion Matrix
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f'Test Confusion Matrix (Accuracy: {test_accuracy:.1%})')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Preprocessor and Datasets
    preprocessor = Preprocessor()
    _, val_test_transform = preprocessor.get_transforms()

    data_path = './Data'
    val_dataset = ResearchDataset(data_path, transform=val_test_transform, mode='Valid')
    test_dataset = ResearchDataset(data_path, transform=val_test_transform, mode='Test')

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Instantiate and load model
    model = ProposedModel(model_size='small', backbone_type='vit', use_cross_attention=True, kp_head_type='hrnet').to(device)
    model_path = '../../Results/best_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")

    class_names = ["Normal", "Large", "Small"] # Assuming this order based on ResearchDataset class_map

    # Get predictions for validation set
    val_true, val_pred = get_predictions(model, val_loader, device)
    val_cm = confusion_matrix(val_true, val_pred)
    
    # Get predictions for test set
    test_true, test_pred = get_predictions(model, test_loader, device)
    test_cm = confusion_matrix(test_true, test_pred)

    # Load accuracies from results.json
    with open('Results/results.json', 'r') as f:
        results_data = json.load(f)
    val_accuracy = results_data['validation']['accuracy']
    test_accuracy = results_data['test']['accuracy']

    # Plot combined confusion matrices
    plot_combined_confusion_matrices(val_cm, test_cm, class_names, val_accuracy, test_accuracy, 'paper/Figures/combined_confusion_matrix.png')
    print("Combined confusion matrix plotted to paper/Figures/combined_confusion_matrix.png")

    print("Confusion matrices generated successfully.")