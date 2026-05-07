import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import torchvision.transforms as transforms
import os
import copy
import scipy.io
from PIL import Image
import timm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error
import json
from datetime import datetime
import torch.nn.functional as F
import random


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


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are logits, targets are class indices
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class LearnableLoss(nn.Module):
    def __init__(self, num_tasks=3):
        super(LearnableLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros((num_tasks)))

    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss


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


class Trainer:
    """Training framework."""
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, loss_fn_kp, loss_fn_cls, loss_fn_vhs, learnable_loss, device, output_dir, ablation_tasks=None, use_learnable_loss=True, patience=20, min_delta=0.001, fixed_loss_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn_kp = loss_fn_kp
        self.loss_fn_cls = loss_fn_cls
        self.loss_fn_vhs = loss_fn_vhs
        self.learnable_loss = learnable_loss
        self.device = device
        self.output_dir = output_dir
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
        self.ablation_tasks = ablation_tasks if ablation_tasks is not None else ['keypoint', 'classification', 'vhs']
        self.use_learnable_loss = use_learnable_loss
        self.patience = patience
        self.min_delta = min_delta
        self.epochs_no_improve = 0
        self.fixed_loss_weights = fixed_loss_weights if fixed_loss_weights is not None else [1.0, 1.0, 1.0] # Default to equal weights
        self.best_model_wts = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_cls_preds = []
        all_cls_labels = []
        for images, keypoints, class_labels, vhs_labels in self.train_loader:
            images, keypoints, class_labels, vhs_labels = images.to(self.device), keypoints.to(self.device), class_labels.to(self.device), vhs_labels.to(self.device)

            self.optimizer.zero_grad()

            kp_preds, cls_logits, vhs_preds = self.model(images)
            
            losses_list = []
            if 'keypoint' in self.ablation_tasks:
                gt_heatmaps = generate_heatmaps(keypoints.cpu().numpy(), kp_preds.shape[2:]).to(self.device)
                losses_list.append(self.loss_fn_kp(kp_preds, gt_heatmaps))
            if 'classification' in self.ablation_tasks:
                losses_list.append(self.loss_fn_cls(cls_logits, class_labels))
            if 'vhs' in self.ablation_tasks:
                losses_list.append(self.loss_fn_vhs(vhs_preds.squeeze(), vhs_labels))

            if self.use_learnable_loss:
                loss = self.learnable_loss(losses_list)
            else:
                # Apply fixed weights
                weighted_losses = [w * l for w, l in zip(self.fixed_loss_weights, losses_list)]
                loss = sum(weighted_losses)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            all_cls_preds.extend(torch.argmax(cls_logits, dim=1).cpu().numpy())
            all_cls_labels.extend(class_labels.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_cls_labels, all_cls_preds)
        return avg_loss, accuracy

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_cls_preds = []
        all_cls_labels = []
        with torch.no_grad():
            for images, keypoints, class_labels, vhs_labels in self.val_loader:
                images, keypoints, class_labels, vhs_labels = images.to(self.device), keypoints.to(self.device), class_labels.to(self.device), vhs_labels.to(self.device)

                kp_preds, cls_logits, vhs_preds = self.model(images)

                losses_list = []
                if 'keypoint' in self.ablation_tasks:
                    gt_heatmaps = generate_heatmaps(keypoints.cpu().numpy(), kp_preds.shape[2:]).to(self.device)
                    losses_list.append(self.loss_fn_kp(kp_preds, gt_heatmaps))
                if 'classification' in self.ablation_tasks:
                    losses_list.append(self.loss_fn_cls(cls_logits, class_labels))
                if 'vhs' in self.ablation_tasks:
                    losses_list.append(self.loss_fn_vhs(vhs_preds.squeeze(), vhs_labels))

                if self.use_learnable_loss:
                    loss = self.learnable_loss(losses_list)
                else:
                    # Apply fixed weights
                    weighted_losses = [w * l for w, l in zip(self.fixed_loss_weights, losses_list)]
                    loss = sum(weighted_losses)

                total_loss += loss.item()

                all_cls_preds.extend(torch.argmax(cls_logits, dim=1).cpu().numpy())
                all_cls_labels.extend(class_labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_cls_labels, all_cls_preds)

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
        return avg_loss, accuracy

    def train(self, num_epochs):
        best_val_loss = 1000
        for epoch in range(num_epochs):
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.validate()
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
  
            if val_loss < (best_val_loss - self.min_delta):
                best_val_loss = val_loss
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                if val_accuracy >= 0.8:
                    print(f'Validation accuracy reached {val_accuracy:.4f}, stopping training.')
                    break                
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve == self.patience:
                    print(f'Early stopping triggered after {epoch+1} epochs. Returning history.')
                    return self.history, self.model

            self.scheduler.step()

        print("Training completed. Returning history.")
        self.model.load_state_dict(self.best_model_wts)
        return self.history, self.model


class Evaluator:
    """Evaluation framework."""
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def evaluate(self):
        self.model.eval()
        all_cls_preds = []
        all_cls_labels = []
        all_vhs_preds = []
        all_vhs_labels = []

        with torch.no_grad():
            for images, keypoints, class_labels, vhs_labels in self.test_loader:
                images, keypoints, class_labels, vhs_labels = images.to(self.device), keypoints.to(self.device), class_labels.to(self.device), vhs_labels.to(self.device)

                _, cls_logits, vhs_preds = self.model(images)

                all_cls_preds.extend(torch.argmax(cls_logits, dim=1).cpu().numpy())
                all_cls_labels.extend(class_labels.cpu().numpy())
                all_vhs_preds.extend(vhs_preds.squeeze().cpu().numpy().flatten())
                all_vhs_labels.extend(vhs_labels.cpu().numpy().flatten())

        accuracy = accuracy_score(all_cls_labels, all_cls_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_cls_labels, all_cls_preds, average='macro')
        mae = mean_absolute_error(all_vhs_labels, all_vhs_preds)

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'mae_vhs': float(mae)
        }


def run_comprehensive_experiments(smoke_test=False, model_size='small', backbone_type='vit', ablation_tasks=None, use_cross_attention=True, kp_head_type='hrnet', use_learnable_loss=True, fixed_loss_weights=None, output_file=None, seed=42, num_epochs=200, load_weights_from=None):
    """Orchestrates the full experimental pipeline."""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file is not None:
        output_dir = os.path.dirname(output_file)
    else:
        output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    preprocessor = Preprocessor()
    train_transform, val_test_transform = preprocessor.get_transforms()

    data_path = '../Dog_data'
    train_dataset = ResearchDataset(data_path, transform=train_transform, mode='Train')
    val_dataset = ResearchDataset(data_path, transform=val_test_transform, mode='Valid')
    test_dataset = ResearchDataset(data_path, transform=val_test_transform, mode='Test')


    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



    model = ProposedModel(model_size=model_size, backbone_type=backbone_type, use_cross_attention=use_cross_attention, kp_head_type=kp_head_type).to(device)
    
    if load_weights_from:
        print(f"Loading weights from {load_weights_from}")
        model.load_state_dict(torch.load(load_weights_from))

    if use_learnable_loss:
        learnable_loss = LearnableLoss().to(device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(learnable_loss.parameters()), lr=1e-4)
    else:
        learnable_loss = None # Not used if use_learnable_loss is False
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    loss_fn_kp = nn.MSELoss()
    loss_fn_cls = FocalLoss()
    loss_fn_vhs = nn.MSELoss()

    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, loss_fn_kp, loss_fn_cls, loss_fn_vhs, learnable_loss, device, output_dir, ablation_tasks=ablation_tasks, use_learnable_loss=use_learnable_loss, fixed_loss_weights=fixed_loss_weights)
    if smoke_test:
        num_epochs = 1

    history, model = trainer.train(num_epochs)
    print("Training finished.")

    print("Starting evaluation on the validation set...")
    val_evaluator = Evaluator(model, val_loader, device)
    val_results = val_evaluator.evaluate()
    print("Validation results:", val_results)

    print("Starting evaluation on the test set...")
    test_evaluator = Evaluator(model, test_loader, device)
    test_results = test_evaluator.evaluate()
    print("Test results:", test_results)

    results = {
        'seed': seed,
        'validation': val_results,
        'test': test_results,
        'history': history
    }
    print("Evaluation finished.")

    if output_file is None:
        output_file = os.path.join(output_dir, 'results.json')

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--model-size', type=str, default='small', choices=['small', 'base'])
    parser.add_argument('--backbone-type', type=str, default='vit', choices=['vit', 'resnet'])
    parser.add_argument('--ablation-tasks', nargs='+', type=str, default=['keypoint', 'classification', 'vhs'], help='List of tasks to include in ablation study (keypoint, classification, vhs)')
    parser.add_argument('--use-cross-attention', action='store_true', default=True)
    parser.add_argument('--kp-head-type', type=str, default='hrnet', choices=['hrnet', 'simple'])
    parser.add_argument('--use-learnable-loss', action='store_true', default=True)
    parser.add_argument('--fixed-loss-weights', nargs='+', type=float, help='List of fixed loss weights for keypoint, classification, and vhs tasks (e.g., 1.0 1.0 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_file', type=str, default='../Results/results.json', help='Path to save the results JSON file')
    parser.add_argument('--num-epochs', type=int, default=200, help='Number of epochs to train for')
    parser.add_argument('--load-weights-from', type=str, default=None, help='Path to load model weights from')
    args = parser.parse_args()

    run_comprehensive_experiments(smoke_test=args.smoke_test, model_size=args.model_size, backbone_type=args.backbone_type, ablation_tasks=args.ablation_tasks, use_cross_attention=args.use_cross_attention, kp_head_type=args.kp_head_type, use_learnable_loss=args.use_learnable_loss, fixed_loss_weights=args.fixed_loss_weights, output_file=args.output_file, seed=args.seed, num_epochs=args.num_epochs, load_weights_from=args.load_weights_from)