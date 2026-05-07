"""
β-VAE Model for Latent-Navigator-Lite Experiment
Implements the VAE with disentangled latent space and 2D projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class BetaVAE(nn.Module):
    """
    β-VAE for learning disentangled representations
    
    Architecture based on the paper specification:
    - Latent dimension: 16
    - β = 4 for disentanglement
    - Works with dSprites, EMNIST, or geometric shapes
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (1, 64, 64),
                 latent_dim: int = 16,
                 hidden_dims: list = [32, 64, 128, 256],
                 beta: float = 4.0):
        super().__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta
        self.hidden_dims = hidden_dims
        
        # Calculate flattened dimension after convolutions
        self.flat_dim = self._calculate_flat_dim()
        
        # Encoder
        self.encoder = self._build_encoder()
        
        # Calculate actual dimensions by running a test forward pass
        with torch.no_grad():
            test_input = torch.zeros(1, *self.input_shape)
            encoded = self.encoder(test_input)
            actual_flat_dim = encoded.numel() // encoded.size(0)
        
        # Latent layers
        self.fc_mu = nn.Linear(actual_flat_dim, latent_dim)
        self.fc_var = nn.Linear(actual_flat_dim, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, actual_flat_dim)
        self.decoder = self._build_decoder()
        
        # Final output layer - add one more upsampling to get back to 64x64
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[0], self.input_shape[0], 
                              kernel_size=4, stride=2, padding=1),  # Upsample to match input size
            nn.Sigmoid()
        )
    
    def _calculate_flat_dim(self) -> int:
        """Calculate flattened dimension after encoder convolutions"""
        x = torch.zeros(1, *self.input_shape)
        
        # Build temporary encoder to calculate dimensions
        temp_modules = []
        in_channels = self.input_shape[0]
        
        for h_dim in self.hidden_dims:
            temp_modules.extend([
                nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ])
            in_channels = h_dim
        
        temp_encoder = nn.Sequential(*temp_modules)
        
        with torch.no_grad():
            x = temp_encoder(x)
        
        return x.size(1) * x.size(2) * x.size(3)  # Include channel dimension
    
    def _build_encoder(self) -> nn.Module:
        """Build encoder network"""
        modules = []
        in_channels = self.input_shape[0]
        
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim
        
        return nn.Sequential(*modules)
    
    def _build_decoder(self) -> nn.Module:
        """Build decoder network"""
        modules = []
        hidden_dims = self.hidden_dims[::-1]  # Reverse for decoder
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                      kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2)
                )
            )
        
        return nn.Sequential(*modules)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent parameters
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            mu: Mean of latent distribution [batch, latent_dim]
            log_var: Log variance of latent distribution [batch, latent_dim]
        """
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        
        return mu, log_var
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output
        
        Args:
            z: Latent representation [batch, latent_dim]
            
        Returns:
            reconstruction: Reconstructed input [batch, channels, height, width]
        """
        h = self.decoder_input(z)
        # Calculate proper dimensions for reshaping
        with torch.no_grad():
            test_input = torch.zeros(1, *self.input_shape)
            encoded = self.encoder(test_input)
            batch_size, channels, height, width = encoded.shape
        
        h = h.view(-1, channels, height, width)
        
        h = self.decoder(h)
        reconstruction = self.final_layer(h)
        
        return reconstruction
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling
        
        Args:
            mu: Mean [batch, latent_dim]
            log_var: Log variance [batch, latent_dim]
            
        Returns:
            z: Sampled latent representation [batch, latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VAE
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            outputs: Dictionary containing reconstruction, mu, log_var, z, and loss
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        
        # Compute VAE loss
        loss = self._compute_vae_loss(reconstruction, x, mu, log_var)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'log_var': log_var,
            'z': z,
            'loss': loss
        }
    
    def _compute_vae_loss(self, reconstruction: torch.Tensor, x: torch.Tensor, 
                         mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Compute β-VAE loss components
        
        Args:
            reconstruction: Reconstructed input
            x: Original input
            mu: Mean from encoder
            log_var: Log variance from encoder
            
        Returns:
            total_loss: Combined reconstruction + KL loss
        """
        # Reconstruction loss (MSE for continuous, BCE for binary)
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total β-VAE loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss
    
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute β-VAE loss
        
        Args:
            x: Input tensor
            outputs: Forward pass outputs
            
        Returns:
            losses: Dictionary containing loss components
        """
        reconstruction = outputs['reconstruction']
        mu = outputs['mu']
        log_var = outputs['log_var']
        
        # Reconstruction loss (MSE for continuous, BCE for binary)
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total β-VAE loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'beta_kl_loss': self.beta * kl_loss
        }
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from the latent space
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            samples: Generated samples [num_samples, channels, height, width]
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two inputs in latent space
        
        Args:
            x1: First input [1, channels, height, width]
            x2: Second input [1, channels, height, width]
            steps: Number of interpolation steps
            
        Returns:
            interpolations: Interpolated samples [steps, channels, height, width]
        """
        with torch.no_grad():
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)
            
            # Linear interpolation in latent space
            alphas = torch.linspace(0, 1, steps, device=x1.device)
            interpolations = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                x_interp = self.decode(z_interp)
                interpolations.append(x_interp)
            
            return torch.cat(interpolations, dim=0)


class ProjectionNetwork(nn.Module):
    """
    Projection network P_φ: R^16 → R^2
    
    Maps 16D latent space to 2D for visualization and navigation
    """
    
    def __init__(self, 
                 input_dim: int = 16,
                 hidden_dim: int = 64,
                 output_dim: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # MLP: 16 → 64 → 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project latent representation to 2D
        
        Args:
            z: Latent representation [batch, input_dim]
            
        Returns:
            projection: 2D projection [batch, output_dim]
        """
        return self.network(z)
    
    def compute_projection_loss(self, 
                               z: torch.Tensor, 
                               target_factors: torch.Tensor,
                               factor_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute projection loss to preserve important factors
        
        Args:
            z: Latent representations [batch, input_dim]
            target_factors: Target factor values [batch, num_factors]
            factor_weights: Weights for different factors [num_factors]
            
        Returns:
            projection_loss: Loss encouraging meaningful 2D projection
        """
        projections = self.forward(z)
        
        if factor_weights is None:
            factor_weights = torch.ones(target_factors.size(1), device=z.device)
        
        # Encourage projection to correlate with important factors
        # Use first two factors for x and y coordinates
        target_2d = target_factors[:, :2]  # Take first two factors
        
        # MSE loss between projection and normalized target factors
        target_2d_norm = 2 * (target_2d - target_2d.min(dim=0)[0]) / (
            target_2d.max(dim=0)[0] - target_2d.min(dim=0)[0] + 1e-8) - 1
        
        projection_loss = F.mse_loss(projections, target_2d_norm)
        
        return projection_loss
    
    def get_2d_grid(self, grid_size: int = 50, bounds: float = 1.0) -> torch.Tensor:
        """
        Generate 2D grid for exploration
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            bounds: Bounds of the grid [-bounds, bounds]
            
        Returns:
            grid_points: Grid points [grid_size^2, 2]
        """
        x = torch.linspace(-bounds, bounds, grid_size)
        y = torch.linspace(-bounds, bounds, grid_size)
        
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        return grid_points


def create_beta_vae(config: Dict) -> BetaVAE:
    """Factory function to create β-VAE"""
    return BetaVAE(
        input_shape=config.get('input_shape', (1, 64, 64)),
        latent_dim=config.get('latent_dim', 16),
        hidden_dims=config.get('hidden_dims', [32, 64, 128, 256]),
        beta=config.get('beta', 4.0)
    )


def create_projection_network(config: Dict) -> ProjectionNetwork:
    """Factory function to create projection network"""
    return ProjectionNetwork(
        input_dim=config.get('latent_dim', 16),
        hidden_dim=config.get('projection_hidden_dim', 64),
        output_dim=config.get('projection_output_dim', 2)
    )
