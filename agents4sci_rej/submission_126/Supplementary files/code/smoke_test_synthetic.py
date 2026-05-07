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

# Import necessary classes and functions from run_experiments.py
# Assuming run_experiments.py is in the same directory or accessible via PYTHONPATH
from run_experiments import FocalLoss, LearnableLoss, Preprocessor, ProposedModel, Trainer, Evaluator, generate_heatmaps

# --- Synthetic ResearchDataset (mimicking the original but generating data) ---
class SyntheticResearchDataset(Dataset):
    """Synthetic Dataset for smoke testing."""
    def __init__(self, transform=None, mode='train', num_samples=10):
        self.transform = transform
        self.mode = mode
        self.num_samples = num_samples
        self.image_size = (224, 224)
        self.num_keypoints = 6
        self.num_classes = 3

        self.labels_data = []
        for _ in range(self.num_samples):
            self.labels_data.append({
                'class': random.randint(0, self.num_classes - 1),
                'keypoints': np.random.rand(self.num_keypoints, 2) * self.image_size[0], # Random keypoints
                'vhs': np.random.rand() * 10 + 5 # Random VHS score
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic image
        image = Image.new('RGB', self.image_size, color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

        keypoints = torch.from_numpy(self.labels_data[idx]['keypoints']).float()
        vhs = torch.tensor(self.labels_data[idx]['vhs']).float().squeeze()
        class_label = self.labels_data[idx]['class']

        if self.transform:
            image = self.transform(image)

        return image, keypoints, class_label, vhs

# --- Synthetic run_comprehensive_experiments function ---
def run_comprehensive_experiments_synthetic(smoke_test=True, model_size='small', backbone_type='vit', ablation_tasks=None, use_cross_attention=True, kp_head_type='hrnet', use_learnable_loss=True, fixed_loss_weights=None, output_dir=None):
    """Orchestrates the experimental pipeline with synthetic data for smoke testing."""
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is not None:
        output_dir = output_dir + f"/results_{timestamp}/"
    else:
        output_dir =  f"results_synthetic_{timestamp}" # Use a different name for synthetic results
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    preprocessor = Preprocessor()
    train_transform, val_test_transform = preprocessor.get_transforms()

    # Use SyntheticResearchDataset
    train_dataset = SyntheticResearchDataset(transform=train_transform, mode='Train', num_samples=10)
    val_dataset = SyntheticResearchDataset(transform=val_test_transform, mode='Valid', num_samples=5)
    test_dataset = SyntheticResearchDataset(transform=val_test_transform, mode='Test', num_samples=5)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True) # Smaller batch size for smoke test
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    model = ProposedModel(model_size=model_size, backbone_type=backbone_type, use_cross_attention=use_cross_attention, kp_head_type=kp_head_type).to(device)
    if use_learnable_loss:
        learnable_loss = LearnableLoss().to(device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(learnable_loss.parameters()), lr=1e-4)
    else:
        learnable_loss = None
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    loss_fn_kp = nn.MSELoss()
    loss_fn_cls = FocalLoss()
    loss_fn_vhs = nn.MSELoss()

    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, loss_fn_kp, loss_fn_cls, loss_fn_vhs, learnable_loss, device, output_dir, ablation_tasks=ablation_tasks, use_learnable_loss=use_learnable_loss, fixed_loss_weights=fixed_loss_weights)
    
    # Force num_epochs to 1 for smoke test
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
        'validation': val_results,
        'test': test_results,
        'history': history
    }
    print("Evaluation finished.")

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print("Results:", results)


if __name__ == '__main__':
    # This script does not use argparse for simplicity,
    # as it's a dedicated synthetic smoke test.
    # All parameters are hardcoded or defaulted for a quick run.
    print("Running synthetic smoke test...")
    run_comprehensive_experiments_synthetic(smoke_test=True)
    print("Synthetic smoke test completed successfully.")
