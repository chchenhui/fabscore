"""
Pathway Signature Encoder with Attention Mechanisms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class PathwayEncoder(nn.Module):
    """
    Pathway signature encoder with multi-head attention for learning
    representations from 32-dimensional pathway activity scores.
    """
    
    def __init__(self, 
                 input_dim: int = 32,
                 hidden_dims: list = [64, 128, 64],
                 output_dim: int = 32,
                 num_attention_heads: int = 4,
                 dropout_rate: float = 0.1,
                 use_pathway_attention: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_pathway_attention = use_pathway_attention
        
        # Build encoding layers
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after last layer
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout_rate))
        
        # Pathway-specific attention mechanism
        if use_pathway_attention:
            self.pathway_attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=num_attention_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            
            # Learnable pathway embeddings
            self.pathway_embeddings = nn.Parameter(
                torch.randn(input_dim, output_dim) * 0.1
            )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through pathway encoder.
        
        Args:
            x: Input pathway signatures [batch_size, 32]
            
        Returns:
            encoded_features: Encoded pathway features [batch_size, output_dim]
            attention_weights: Attention weights if attention is used
        """
        batch_size = x.size(0)
        
        # Pass through encoding layers
        encoded = x
        for layer in self.layers:
            encoded = layer(encoded)
        
        attention_weights = None
        if self.use_pathway_attention:
            # Add pathway position embeddings
            pathway_positions = torch.arange(self.input_dim, device=x.device)
            position_embeds = self.pathway_embeddings[pathway_positions]
            position_embeds = position_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Reshape for attention: [batch_size, num_pathways, feature_dim]
            encoded_reshaped = encoded.unsqueeze(1).expand(-1, self.input_dim, -1)
            encoded_with_pos = encoded_reshaped + position_embeds
            
            # Self-attention across pathway dimensions
            attended_features, attention_weights = self.pathway_attention(
                encoded_with_pos, encoded_with_pos, encoded_with_pos
            )
            
            # Pool attended features
            encoded = attended_features.mean(dim=1)
        
        # Layer normalization
        encoded = self.layer_norm(encoded)
        
        return encoded, attention_weights


class HierarchicalAttention(nn.Module):
    """
    Cross-level attention mechanism for hierarchical classification.
    """
    
    def __init__(self, feature_dim: int = 32, num_levels: int = 3):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_levels = num_levels
        
        # Cross-level attention
        self.cross_level_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Level embeddings for hierarchical positioning
        self.level_embeddings = nn.Embedding(num_levels, feature_dim)
        
        # Level-specific projections
        self.level_projections = nn.ModuleDict({
            'organ': nn.Linear(feature_dim, feature_dim),
            'histology': nn.Linear(feature_dim, feature_dim),
            'molecular': nn.Linear(feature_dim, feature_dim)
        })
        
    def forward(self, features: torch.Tensor, level: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply hierarchical attention for specific level.
        
        Args:
            features: Input features [batch_size, feature_dim]
            level: Hierarchy level ('organ', 'histology', 'molecular')
            
        Returns:
            attended_features: Features after attention
            attention_weights: Attention weights
        """
        batch_size = features.size(0)
        
        # Get level index
        level_mapping = {'organ': 0, 'histology': 1, 'molecular': 2}
        level_idx = torch.tensor(level_mapping[level], device=features.device)
        
        # Add level embeddings
        level_emb = self.level_embeddings(level_idx)
        level_emb = level_emb.unsqueeze(0).expand(batch_size, -1)
        features_with_level = features + level_emb
        
        # Apply level-specific projection
        projected_features = self.level_projections[level](features_with_level)
        
        # Prepare for attention (add sequence dimension)
        query = projected_features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        key_value = features_with_level.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # Cross-level attention
        attended_features, attention_weights = self.cross_level_attention(
            query, key_value, key_value
        )
        
        # Remove sequence dimension
        attended_features = attended_features.squeeze(1)
        
        return attended_features, attention_weights


class PathwayImportanceAnalyzer:
    """
    Analyzer for computing pathway importance using various methods.
    """
    
    def __init__(self, model: nn.Module, pathway_names: list):
        self.model = model
        self.pathway_names = pathway_names
        
    def integrated_gradients(self, 
                           data: torch.Tensor, 
                           target_class: Optional[torch.Tensor] = None,
                           n_steps: int = 50,
                           level: str = 'molecular') -> torch.Tensor:
        """
        Compute integrated gradients for pathway importance.
        
        Args:
            data: Input pathway data [batch_size, 32]
            target_class: Target classes for gradient computation
            n_steps: Number of integration steps
            level: Classification level to analyze
            
        Returns:
            pathway_importance: Importance scores for each pathway
        """
        self.model.eval()
        
        # Create baseline (zero input)
        baseline = torch.zeros_like(data)
        
        # If no target class provided, use predicted class
        if target_class is None:
            with torch.no_grad():
                predictions, _ = self.model(data)
                target_class = predictions[level].argmax(dim=1)
        
        importance_scores = []
        
        for step in range(n_steps):
            alpha = step / n_steps
            interpolated = baseline + alpha * (data - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            predictions, _ = self.model(interpolated)
            
            # Get target logits
            target_logits = predictions[level][range(len(target_class)), target_class]
            
            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=target_logits.sum(),
                inputs=interpolated,
                retain_graph=True,
                create_graph=False
            )[0]
            
            importance_scores.append(gradients)
        
        # Integrate gradients
        integrated_grads = torch.stack(importance_scores).mean(dim=0)
        pathway_importance = (data - baseline) * integrated_grads
        
        return pathway_importance.abs()
    
    def permutation_importance(self, 
                             data: torch.Tensor, 
                             labels: torch.Tensor,
                             level: str = 'molecular',
                             n_permutations: int = 10) -> torch.Tensor:
        """
        Compute permutation importance for pathways.
        
        Args:
            data: Input pathway data
            labels: True labels
            level: Classification level
            n_permutations: Number of permutation iterations
            
        Returns:
            importance_scores: Importance score for each pathway
        """
        self.model.eval()
        
        # Get baseline accuracy
        with torch.no_grad():
            predictions, _ = self.model(data)
            baseline_acc = (predictions[level].argmax(dim=1) == labels).float().mean()
        
        importance_scores = []
        
        for pathway_idx in range(data.size(1)):
            pathway_importance = 0
            
            for _ in range(n_permutations):
                # Permute specific pathway
                permuted_data = data.clone()
                perm_indices = torch.randperm(data.size(0))
                permuted_data[:, pathway_idx] = data[perm_indices, pathway_idx]
                
                # Compute accuracy with permuted pathway
                with torch.no_grad():
                    predictions, _ = self.model(permuted_data)
                    permuted_acc = (predictions[level].argmax(dim=1) == labels).float().mean()
                
                # Importance is decrease in accuracy
                pathway_importance += baseline_acc - permuted_acc
            
            importance_scores.append(pathway_importance / n_permutations)
        
        return torch.tensor(importance_scores)