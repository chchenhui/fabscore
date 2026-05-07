"""
Data Loading for Latent-Navigator-Lite
Supports dSprites, EMNIST, and geometric shapes datasets
"""

import torch
import torch.utils.data as data
import numpy as np
import os
from typing import Tuple, Dict, List, Optional, Any
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import requests
import zipfile
import h5py


class dSpritesDataset(data.Dataset):
    """
    dSprites dataset loader
    
    Dataset contains sprites with factors:
    - Shape: 3 values (square, ellipse, heart)
    - Scale: 6 values
    - Orientation: 40 values
    - Position X: 32 values
    - Position Y: 32 values
    """
    
    def __init__(self, 
                 root: str = './data',
                 download: bool = True,
                 transform: Optional[transforms.Compose] = None):
        self.root = root
        self.transform = transform
        
        if download:
            self._download()
        
        self._load_data()
    
    def _download(self):
        """Download dSprites dataset if not present"""
        os.makedirs(self.root, exist_ok=True)
        
        data_path = os.path.join(self.root, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        
        if not os.path.exists(data_path):
            print("Downloading dSprites dataset...")
            url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
            
            response = requests.get(url)
            with open(data_path, 'wb') as f:
                f.write(response.content)
            
            print("Download completed.")
    
    def _load_data(self):
        """Load dSprites data"""
        data_path = os.path.join(self.root, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        
        with np.load(data_path, allow_pickle=True, encoding='bytes') as data:
            self.images = data['imgs']
            self.latents_values = data['latents_values']
            self.latents_classes = data['latents_classes']
            self.metadata = data['metadata'].item()
        
        # Convert to torch tensors
        self.images = torch.from_numpy(self.images).float().unsqueeze(1)  # Add channel dim
        self.latents_values = torch.from_numpy(self.latents_values).float()
        self.latents_classes = torch.from_numpy(self.latents_classes).long()
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self.images[idx]
        latent_values = self.latents_values[idx]
        latent_classes = self.latents_classes[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'latent_values': latent_values,
            'latent_classes': latent_classes,
            'factors': latent_values[1:]  # Exclude color (always 1)
        }


class GeometricShapesDataset(data.Dataset):
    """
    Synthetic geometric shapes dataset
    
    Generates shapes with controllable factors:
    - Shape: 4 types (circle, square, triangle, diamond)
    - Size: continuous [0.2, 0.8]
    - Rotation: continuous [0, 2π]
    - Position X: continuous [0, 1]
    - Position Y: continuous [0, 1]
    - Color: 3 values (if colored)
    """
    
    def __init__(self,
                 num_samples: int = 50000,
                 image_size: int = 64,
                 colored: bool = False,
                 transform: Optional[transforms.Compose] = None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.colored = colored
        self.transform = transform
        
        # Factor ranges
        self.factor_names = ['shape', 'size', 'rotation', 'pos_x', 'pos_y']
        if colored:
            self.factor_names.append('color')
        
        self.shape_types = ['circle', 'square', 'triangle', 'diamond']
        self.colors = ['red', 'green', 'blue'] if colored else ['white']
        
        # Generate dataset
        self._generate_dataset()
    
    def _generate_dataset(self):
        """Generate synthetic dataset"""
        self.images = []
        self.factors = []
        
        for _ in range(self.num_samples):
            # Sample factors
            shape_idx = np.random.randint(0, len(self.shape_types))
            size = np.random.uniform(0.2, 0.8)
            rotation = np.random.uniform(0, 2 * np.pi)
            pos_x = np.random.uniform(0.1, 0.9)
            pos_y = np.random.uniform(0.1, 0.9)
            color_idx = np.random.randint(0, len(self.colors)) if self.colored else 0
            
            # Generate image
            image = self._draw_shape(
                shape_idx, size, rotation, pos_x, pos_y, color_idx
            )
            
            # Store data
            self.images.append(image)
            
            factor_values = [shape_idx, size, rotation, pos_x, pos_y]
            if self.colored:
                factor_values.append(color_idx)
            
            self.factors.append(factor_values)
        
        # Convert to tensors
        self.images = torch.stack(self.images)
        self.factors = torch.tensor(self.factors, dtype=torch.float32)
    
    def _draw_shape(self, 
                   shape_idx: int, 
                   size: float, 
                   rotation: float, 
                   pos_x: float, 
                   pos_y: float, 
                   color_idx: int) -> torch.Tensor:
        """Draw a shape with given parameters"""
        # Create PIL image
        if self.colored:
            img = Image.new('RGB', (self.image_size, self.image_size), 'black')
        else:
            img = Image.new('L', (self.image_size, self.image_size), 0)
        
        draw = ImageDraw.Draw(img)
        
        # Calculate shape parameters
        center_x = pos_x * self.image_size
        center_y = pos_y * self.image_size
        radius = size * self.image_size * 0.3
        
        color = self.colors[color_idx] if self.colored else 255
        
        # Draw shape
        if self.shape_types[shape_idx] == 'circle':
            bbox = [center_x - radius, center_y - radius, 
                   center_x + radius, center_y + radius]
            draw.ellipse(bbox, fill=color)
        
        elif self.shape_types[shape_idx] == 'square':
            # Create rotated square
            corners = self._get_rotated_square(center_x, center_y, radius, rotation)
            draw.polygon(corners, fill=color)
        
        elif self.shape_types[shape_idx] == 'triangle':
            corners = self._get_rotated_triangle(center_x, center_y, radius, rotation)
            draw.polygon(corners, fill=color)
        
        elif self.shape_types[shape_idx] == 'diamond':
            corners = self._get_rotated_diamond(center_x, center_y, radius, rotation)
            draw.polygon(corners, fill=color)
        
        # Convert to tensor
        img_array = np.array(img)
        if len(img_array.shape) == 2:  # Grayscale
            img_tensor = torch.from_numpy(img_array).float().unsqueeze(0) / 255.0
        else:  # RGB
            img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1) / 255.0
        
        return img_tensor
    
    def _get_rotated_square(self, cx: float, cy: float, size: float, rotation: float) -> List[Tuple[float, float]]:
        """Get corners of rotated square"""
        corners = [(-size, -size), (size, -size), (size, size), (-size, size)]
        rotated_corners = []
        
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        
        for x, y in corners:
            new_x = x * cos_r - y * sin_r + cx
            new_y = x * sin_r + y * cos_r + cy
            rotated_corners.append((new_x, new_y))
        
        return rotated_corners
    
    def _get_rotated_triangle(self, cx: float, cy: float, size: float, rotation: float) -> List[Tuple[float, float]]:
        """Get corners of rotated triangle"""
        corners = [(0, -size), (-size * 0.866, size * 0.5), (size * 0.866, size * 0.5)]
        rotated_corners = []
        
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        
        for x, y in corners:
            new_x = x * cos_r - y * sin_r + cx
            new_y = x * sin_r + y * cos_r + cy
            rotated_corners.append((new_x, new_y))
        
        return rotated_corners
    
    def _get_rotated_diamond(self, cx: float, cy: float, size: float, rotation: float) -> List[Tuple[float, float]]:
        """Get corners of rotated diamond"""
        corners = [(0, -size), (size, 0), (0, size), (-size, 0)]
        rotated_corners = []
        
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        
        for x, y in corners:
            new_x = x * cos_r - y * sin_r + cx
            new_y = x * sin_r + y * cos_r + cy
            rotated_corners.append((new_x, new_y))
        
        return rotated_corners
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self.images[idx]
        factors = self.factors[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'factors': factors,
            'latent_values': factors,  # For compatibility
            'latent_classes': factors.long()  # For compatibility
        }


class ScoringOracle:
    """
    Scoring oracle for latent navigation task
    
    Provides rewards based on hidden factor combinations
    """
    
    def __init__(self, 
                 factor_weights: Optional[Dict[str, float]] = None,
                 target_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        self.factor_weights = factor_weights or {
            'shape': 0.3,
            'size': 0.2,
            'rotation': 0.1,
            'pos_x': 0.2,
            'pos_y': 0.2
        }
        
        self.target_ranges = target_ranges or {
            'shape': (1.0, 2.0),  # Prefer ellipse/triangle
            'size': (0.6, 0.8),   # Larger sizes
            'rotation': (0.5, 1.5),  # Specific rotation range
            'pos_x': (0.3, 0.7),  # Center region
            'pos_y': (0.3, 0.7)   # Center region
        }
    
    def compute_score(self, factors: torch.Tensor) -> torch.Tensor:
        """
        Compute score for given factors
        
        Args:
            factors: Factor values [batch, num_factors] or [num_factors]
            
        Returns:
            scores: Scores [batch] or scalar
        """
        if factors.dim() == 1:
            factors = factors.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = factors.size(0)
        scores = torch.zeros(batch_size)
        
        factor_names = list(self.factor_weights.keys())
        
        for i, factor_name in enumerate(factor_names):
            if i >= factors.size(1):
                continue
            
            factor_values = factors[:, i]
            weight = self.factor_weights[factor_name]
            target_min, target_max = self.target_ranges[factor_name]
            
            # Score based on proximity to target range
            in_range = (factor_values >= target_min) & (factor_values <= target_max)
            
            # Distance-based scoring for continuous factors
            if factor_name in ['size', 'rotation', 'pos_x', 'pos_y']:
                target_center = (target_min + target_max) / 2
                target_width = target_max - target_min
                
                distances = torch.abs(factor_values - target_center)
                factor_scores = torch.clamp(1.0 - distances / target_width, 0.0, 1.0)
            else:
                # Discrete factors (shape)
                factor_scores = in_range.float()
            
            scores += weight * factor_scores
        
        if squeeze_output:
            scores = scores.squeeze(0)
        
        return scores
    
    def get_optimal_factors(self) -> Dict[str, float]:
        """Get optimal factor values for maximum score"""
        optimal_factors = {}
        
        for factor_name, (target_min, target_max) in self.target_ranges.items():
            optimal_factors[factor_name] = (target_min + target_max) / 2
        
        return optimal_factors


class DatasetFactory:
    """Factory for creating datasets"""
    
    @staticmethod
    def create_dataset(dataset_name: str, **kwargs) -> data.Dataset:
        """
        Create dataset by name
        
        Args:
            dataset_name: Name of dataset ('dsprites', 'geometric_shapes', 'emnist')
            **kwargs: Additional arguments for dataset
            
        Returns:
            dataset: Created dataset
        """
        if dataset_name.lower() == 'dsprites':
            return dSpritesDataset(**kwargs)
        
        elif dataset_name.lower() == 'geometric_shapes':
            return GeometricShapesDataset(**kwargs)
        
        elif dataset_name.lower() == 'emnist':
            # Placeholder for EMNIST - would need additional implementation
            raise NotImplementedError("EMNIST dataset not implemented yet")
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    @staticmethod
    def create_scoring_oracle(dataset_name: str, **kwargs) -> ScoringOracle:
        """Create scoring oracle for dataset"""
        if dataset_name.lower() in ['dsprites', 'geometric_shapes']:
            return ScoringOracle(**kwargs)
        else:
            raise ValueError(f"No scoring oracle for dataset: {dataset_name}")


def create_dataset(config: Dict[str, Any]) -> data.Dataset:
    """Factory function to create dataset from config"""
    dataset_name = config.get('dataset_name', 'geometric_shapes')
    dataset_kwargs = config.get('dataset_kwargs', {})
    
    return DatasetFactory.create_dataset(dataset_name, **dataset_kwargs)


def create_data_loader(dataset: data.Dataset, 
                      batch_size: int = 64,
                      shuffle: bool = True,
                      num_workers: int = 0) -> data.DataLoader:
    """Create data loader from dataset"""
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
