# CIFAR-10 data loading and probe set sampling.
# Provides train/test datasets/loaders and a fixed probe subset for controller measurement.

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_transform():
    return T.Compose([T.ToTensor(), T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)])


def get_datasets(data_root="./data"):
    transform = get_transform()
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def get_train_loader(train_dataset, batch_size=128, num_workers=2, seed=None):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        drop_last=True,
    )


def sample_probe_set(test_dataset, probe_size=16, seed=0):
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(test_dataset), size=probe_size, replace=False)
    probe_subset = Subset(test_dataset, indices.tolist())
    images, labels = zip(*[probe_subset[i] for i in range(len(probe_subset))])
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels
