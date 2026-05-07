"""
Representation Mapper T_ψ for BiCA
Maps human representations to AI latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.stats import wasserstein_distance


class RepresentationMapper(nn.Module):
    """
    Representation mapper T_ψ: Z_H -> Z_A
    
    Maps human latent representations to AI latent space.
    Used to minimize representation gap in BiCA objective.
    """
    
    def __init__(self,
                 human_latent_dim: int = 128,
                 ai_latent_dim: int = 128,
                 hidden_dim: int = 64):
        super().__init__()
        
        self.human_latent_dim = human_latent_dim
        self.ai_latent_dim = ai_latent_dim
        self.hidden_dim = hidden_dim
        
        # 2-layer MLP mapper
        self.mapper = nn.Sequential(
            nn.Linear(human_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ai_latent_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, human_latent: torch.Tensor) -> torch.Tensor:
        """
        Map human latent to AI latent space
        
        Args:
            human_latent: [batch, human_latent_dim] human representations
            
        Returns:
            mapped_latent: [batch, ai_latent_dim] mapped representations
        """
        return self.mapper(human_latent)
    
    def compute_alignment_loss(self, 
                              human_latent: torch.Tensor, 
                              ai_latent: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment loss between mapped human and AI latents
        
        Args:
            human_latent: [batch, human_latent_dim] human representations
            ai_latent: [batch, ai_latent_dim] AI representations
            
        Returns:
            alignment_loss: MSE loss between mapped and target representations
        """
        mapped_human = self.forward(human_latent)
        return F.mse_loss(mapped_human, ai_latent)


class RepresentationGapComputer:
    """
    Computes representation gap using Wasserstein distance and CCA correlation
    """
    
    def __init__(self, cca_components: int = 10):
        self.cca_components = cca_components
        self.cca_model = None
    
    def compute_wasserstein_distance(self, 
                                   human_reps: torch.Tensor,
                                   mapped_reps: torch.Tensor) -> float:
        """
        Compute 2-Wasserstein distance between distributions
        
        Args:
            human_reps: [batch, dim] human representations
            mapped_reps: [batch, dim] mapped representations
            
        Returns:
            wasserstein_dist: Wasserstein distance
        """
        # Convert to numpy for scipy
        human_np = human_reps.detach().cpu().numpy()
        mapped_np = mapped_reps.detach().cpu().numpy()
        
        # Compute distance for each dimension and average
        distances = []
        for i in range(human_np.shape[1]):
            dist = wasserstein_distance(human_np[:, i], mapped_np[:, i])
            distances.append(dist)
        
        return np.mean(distances)
    
    def compute_cca_correlation(self, 
                               human_reps: torch.Tensor,
                               ai_reps: torch.Tensor) -> float:
        """
        Compute CCA correlation between human and AI representations
        
        Args:
            human_reps: [batch, human_dim] human representations
            ai_reps: [batch, ai_dim] AI representations
            
        Returns:
            cca_corr: Average CCA correlation
        """
        # Convert to numpy
        human_np = human_reps.detach().cpu().numpy()
        ai_np = ai_reps.detach().cpu().numpy()
        
        # Ensure we have enough samples
        if human_np.shape[0] < 2:
            return 0.0
        
        # Fit CCA
        n_components = min(self.cca_components, 
                          human_np.shape[0] - 1,
                          human_np.shape[1],
                          ai_np.shape[1])
        
        if n_components < 1:
            return 0.0
        
        try:
            cca = CCA(n_components=n_components)
            human_canonical, ai_canonical = cca.fit_transform(human_np, ai_np)
            
            # Compute correlations between canonical components
            correlations = []
            for i in range(n_components):
                corr = np.corrcoef(human_canonical[:, i], ai_canonical[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            print(f"CCA computation failed: {e}")
            return 0.0
    
    def compute_representation_gap(self,
                                  human_reps: torch.Tensor,
                                  ai_reps: torch.Tensor,
                                  mapper: RepresentationMapper) -> Dict[str, float]:
        """
        Compute full representation gap metric
        
        Args:
            human_reps: [batch, human_dim] human representations
            ai_reps: [batch, ai_dim] AI representations
            mapper: Trained representation mapper
            
        Returns:
            gap_metrics: Dictionary with gap components
        """
        # Map human representations
        mapped_human = mapper(human_reps)
        
        # Compute Wasserstein distance
        wasserstein_dist = self.compute_wasserstein_distance(human_reps, mapped_human)
        
        # Compute CCA correlation
        cca_corr = self.compute_cca_correlation(mapped_human, ai_reps)
        
        # Combined representation gap
        rep_gap = wasserstein_dist + (1 - cca_corr)
        
        return {
            'wasserstein_distance': wasserstein_dist,
            'cca_correlation': cca_corr,
            'representation_gap': rep_gap
        }


class LatentExtractor:
    """
    Helper class to extract latent representations from models
    """
    
    def __init__(self):
        self.human_latents = []
        self.ai_latents = []
    
    def extract_human_latent(self, human_model, human_obs, ai_message, instructor_action, hidden):
        """Extract human latent representation"""
        with torch.no_grad():
            # Use GRU hidden state as the latent representation
            # This ensures consistent dimensionality with human_gru_hidden
            if hasattr(human_model, 'gru') and hidden is not None:
                # Use the GRU hidden state as latent
                latent = hidden.squeeze(0)  # Remove sequence dimension
            else:
                # Fallback: use encoded observation
                obs_encoded = human_model.obs_encoder(human_obs)
                latent = obs_encoded
            return latent
    
    def extract_ai_latent(self, ai_model, ai_obs, human_message, hidden):
        """Extract AI latent representation"""
        with torch.no_grad():
            # Use policy's internal representation as latent
            # This should match policy_hidden_dim
            msg_embed = ai_model.message_embed(human_message)
            input_features = torch.cat([ai_obs, msg_embed], dim=-1)
            
            # Get the encoded features (should match policy_hidden_dim)
            if hasattr(ai_model, 'policy_net'):
                # Use policy network's hidden representation
                latent = ai_model.policy_net[0](input_features)  # First layer output
            else:
                # Fallback: use observation encoder
                latent = ai_model.obs_encoder(input_features)
            return latent
    
    def collect_latents(self, human_latent, ai_latent):
        """Collect latent representations for batch processing"""
        self.human_latents.append(human_latent.detach().cpu())
        self.ai_latents.append(ai_latent.detach().cpu())
    
    def get_batch_latents(self):
        """Get collected latents as batched tensors"""
        if not self.human_latents or not self.ai_latents:
            return None, None
        
        human_batch = torch.cat(self.human_latents, dim=0)
        ai_batch = torch.cat(self.ai_latents, dim=0)
        
        return human_batch, ai_batch
    
    def clear(self):
        """Clear collected latents"""
        self.human_latents.clear()
        self.ai_latents.clear()


def create_representation_mapper(config: Dict) -> RepresentationMapper:
    """Factory function to create representation mapper"""
    # Calculate actual latent dimensions based on model architecture
    # Human latent is based on human GRU hidden dimension
    human_latent_dim = config.get('human_gru_hidden', 128)
    
    # AI latent is based on policy hidden dimension  
    ai_latent_dim = config.get('policy_hidden_dim', 256)
    
    return RepresentationMapper(
        human_latent_dim=human_latent_dim,
        ai_latent_dim=ai_latent_dim,
        hidden_dim=config.get('mapper_hidden_dim', 64)
    )


def create_gap_computer(config: Dict) -> RepresentationGapComputer:
    """Factory function to create representation gap computer"""
    return RepresentationGapComputer(
        cca_components=config.get('cca_components', 10)
    )


def create_latent_extractor() -> LatentExtractor:
    """Factory function to create latent extractor"""
    return LatentExtractor()
