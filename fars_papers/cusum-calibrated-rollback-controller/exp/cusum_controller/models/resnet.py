# ResNet-18 wrapper for CIFAR-10 (10-class, no pretrained weights).
# Includes probe_loss utility for evaluating on a fixed probe set.

import torch
import torch.nn as nn
import torchvision.models


def create_resnet18(num_classes=10):
    return torchvision.models.resnet18(weights=None, num_classes=num_classes)


def probe_loss(model, probe_images, probe_labels, device):
    model.eval()
    with torch.no_grad():
        images = probe_images.to(device)
        labels = probe_labels.to(device)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
    model.train()
    return loss.item()
