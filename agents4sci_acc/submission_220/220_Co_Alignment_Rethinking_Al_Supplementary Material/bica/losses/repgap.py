"""
Representation Gap Loss for BiCA
Implements Wasserstein distance + CCA correlation loss
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.cross_decomposition import CCA
import ot  # Python Optimal Transport library


class RepresentationGapLoss:
    """
    Representation gap loss combining Wasserstein distance and CCA correlation
    
    RepGap_ψ = W_2^2(P(z^H), P(T_ψ(z^H))) + (1 - ρ_CCA)
    """
    
    def __init__(self, 
                 mu: float = 0.1,
                 cca_components: int = 10,
                 use_sinkhorn: bool = True,
                 sinkhorn_reg: float = 0.1):
        self.mu = mu
        self.cca_components = cca_components
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_reg = sinkhorn_reg
    
    def compute_wasserstein_loss(self,
                                human_reps: torch.Tensor,
                                mapped_reps: torch.Tensor) -> torch.Tensor:
        """
        Compute 2-Wasserstein distance loss between distributions
        
        Args:
            human_reps: [batch, dim] human representations
            mapped_reps: [batch, dim] mapped human representations
            
        Returns:
            wasserstein_loss: Wasserstein distance loss
        """
        if self.use_sinkhorn:
            return self._compute_sinkhorn_loss(human_reps, mapped_reps)
        else:
            return self._compute_exact_wasserstein_loss(human_reps, mapped_reps)
    
    def _compute_sinkhorn_loss(self,
                              human_reps: torch.Tensor,
                              mapped_reps: torch.Tensor) -> torch.Tensor:
        """
        Compute Sinkhorn approximation of Wasserstein distance
        
        Args:
            human_reps: [batch, dim] human representations
            mapped_reps: [batch, dim] mapped representations
            
        Returns:
            sinkhorn_loss: Sinkhorn divergence
        """
        # Check if dimensions match, if not, project to common dimension
        if human_reps.size(-1) != mapped_reps.size(-1):
            # Project both to the smaller dimension for fair comparison
            min_dim = min(human_reps.size(-1), mapped_reps.size(-1))
            human_reps = human_reps[:, :min_dim]
            mapped_reps = mapped_reps[:, :min_dim]
        
        # Compute cost matrix (squared Euclidean distance)
        cost_matrix = torch.cdist(human_reps, mapped_reps, p=2) ** 2
        
        # Uniform distributions
        a = torch.ones(human_reps.size(0), device=human_reps.device) / human_reps.size(0)
        b = torch.ones(mapped_reps.size(0), device=mapped_reps.device) / mapped_reps.size(0)
        
        # Sinkhorn algorithm
        sinkhorn_loss = self._sinkhorn_divergence(cost_matrix, a, b)
        
        return sinkhorn_loss
    
    def _sinkhorn_divergence(self,
                           cost_matrix: torch.Tensor,
                           a: torch.Tensor,
                           b: torch.Tensor,
                           num_iters: int = 100) -> torch.Tensor:
        """
        Sinkhorn algorithm for optimal transport
        
        Args:
            cost_matrix: [n, m] cost matrix
            a: [n] source distribution
            b: [m] target distribution
            num_iters: Number of Sinkhorn iterations
            
        Returns:
            sinkhorn_distance: Sinkhorn divergence
        """
        # Initialize dual variables
        u = torch.zeros_like(a)
        v = torch.zeros_like(b)
        
        # Sinkhorn iterations
        for _ in range(num_iters):
            u = self.sinkhorn_reg * (torch.log(a + 1e-8) - torch.logsumexp(
                (u.unsqueeze(1) + v.unsqueeze(0) - cost_matrix) / self.sinkhorn_reg, dim=1
            ))
            v = self.sinkhorn_reg * (torch.log(b + 1e-8) - torch.logsumexp(
                (u.unsqueeze(1) + v.unsqueeze(0) - cost_matrix) / self.sinkhorn_reg, dim=0
            ))
        
        # Compute transport plan
        transport_plan = torch.exp((u.unsqueeze(1) + v.unsqueeze(0) - cost_matrix) / self.sinkhorn_reg)
        
        # Sinkhorn cost
        sinkhorn_cost = (transport_plan * cost_matrix).sum()
        
        return sinkhorn_cost
    
    def _compute_exact_wasserstein_loss(self,
                                      human_reps: torch.Tensor,
                                      mapped_reps: torch.Tensor) -> torch.Tensor:
        """
        Compute exact Wasserstein distance using scipy (slower but exact)
        
        Args:
            human_reps: [batch, dim] human representations
            mapped_reps: [batch, dim] mapped representations
            
        Returns:
            wasserstein_loss: Exact Wasserstein distance
        """
        # Convert to numpy
        human_np = human_reps.detach().cpu().numpy()
        mapped_np = mapped_reps.detach().cpu().numpy()
        
        # Compute Wasserstein distance for each dimension
        distances = []
        for i in range(human_np.shape[1]):
            dist = wasserstein_distance(human_np[:, i], mapped_np[:, i])
            distances.append(dist ** 2)  # Square for W_2^2
        
        # Average over dimensions
        wasserstein_loss = torch.tensor(np.mean(distances), 
                                      device=human_reps.device, 
                                      dtype=human_reps.dtype)
        
        return wasserstein_loss
    
    def compute_cca_loss(self,
                        human_reps: torch.Tensor,
                        ai_reps: torch.Tensor) -> torch.Tensor:
        """
        Compute CCA correlation loss
        
        Args:
            human_reps: [batch, human_dim] human representations
            ai_reps: [batch, ai_dim] AI representations
            
        Returns:
            cca_loss: (1 - ρ_CCA) loss
        """
        # Handle dimension mismatch by projecting to common dimension
        if human_reps.size(-1) != ai_reps.size(-1):
            min_dim = min(human_reps.size(-1), ai_reps.size(-1))
            human_reps = human_reps[:, :min_dim]
            ai_reps = ai_reps[:, :min_dim]
        
        # Convert to numpy for sklearn CCA
        human_np = human_reps.detach().cpu().numpy()
        ai_np = ai_reps.detach().cpu().numpy()
        
        # Check if we have enough samples
        batch_size = human_np.shape[0]
        if batch_size < 2:
            return torch.tensor(1.0, device=human_reps.device)  # Maximum loss
        
        # Determine number of components
        n_components = min(self.cca_components, 
                          batch_size - 1,
                          human_np.shape[1],
                          ai_np.shape[1])
        
        if n_components < 1:
            return torch.tensor(1.0, device=human_reps.device)
        
        try:
            # Fit CCA
            cca = CCA(n_components=n_components)
            human_canonical, ai_canonical = cca.fit_transform(human_np, ai_np)
            
            # Compute correlations between canonical components
            correlations = []
            for i in range(n_components):
                corr = np.corrcoef(human_canonical[:, i], ai_canonical[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            # Average correlation
            if correlations:
                avg_correlation = np.mean(correlations)
            else:
                avg_correlation = 0.0
            
            # CCA loss (1 - correlation)
            cca_loss = 1.0 - avg_correlation
            
        except Exception as e:
            print(f"CCA computation failed: {e}")
            cca_loss = 1.0  # Maximum loss if CCA fails
        
        return torch.tensor(cca_loss, device=human_reps.device, dtype=human_reps.dtype)
    
    def compute_differentiable_cca_loss(self,
                                      human_reps: torch.Tensor,
                                      ai_reps: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable approximation of CCA correlation
        
        Args:
            human_reps: [batch, human_dim] human representations
            ai_reps: [batch, ai_dim] AI representations
            
        Returns:
            cca_loss: Differentiable CCA approximation loss
        """
        # Handle dimension mismatch by projecting to common dimension
        if human_reps.size(-1) != ai_reps.size(-1):
            min_dim = min(human_reps.size(-1), ai_reps.size(-1))
            human_reps = human_reps[:, :min_dim]
            ai_reps = ai_reps[:, :min_dim]
        
        # Center the representations
        human_centered = human_reps - human_reps.mean(dim=0, keepdim=True)
        ai_centered = ai_reps - ai_reps.mean(dim=0, keepdim=True)
        
        # Compute covariance matrices
        batch_size = human_reps.size(0)
        
        # Cross-covariance
        cross_cov = torch.mm(human_centered.t(), ai_centered) / (batch_size - 1)
        
        # Auto-covariances
        human_cov = torch.mm(human_centered.t(), human_centered) / (batch_size - 1)
        ai_cov = torch.mm(ai_centered.t(), ai_centered) / (batch_size - 1)
        
        # Add regularization for numerical stability
        reg = 1e-4
        human_cov += reg * torch.eye(human_cov.size(0), device=human_cov.device)
        ai_cov += reg * torch.eye(ai_cov.size(0), device=ai_cov.device)
        
        # Compute CCA objective (trace of correlation matrix)
        try:
            # Cholesky decomposition for efficiency
            L_h = torch.linalg.cholesky(human_cov)
            L_a = torch.linalg.cholesky(ai_cov)
            
            # Solve linear systems
            temp1 = torch.linalg.solve(L_h, cross_cov)
            temp2 = torch.linalg.solve(L_a, cross_cov.t())
            
            # Correlation matrix
            corr_matrix = torch.mm(temp1.t(), temp2)
            
            # Trace (sum of canonical correlations)
            trace_corr = torch.trace(corr_matrix)
            
            # Normalize by minimum dimension
            min_dim = min(human_reps.size(1), ai_reps.size(1))
            normalized_corr = trace_corr / min_dim
            
            # CCA loss (1 - normalized correlation)
            cca_loss = 1.0 - torch.clamp(normalized_corr, 0.0, 1.0)
            
        except Exception as e:
            # Fallback to simple correlation if numerical issues
            print(f"Differentiable CCA failed, using simple correlation: {e}")
            
            # Simple correlation fallback
            human_flat = human_centered.view(batch_size, -1)
            ai_flat = ai_centered.view(batch_size, -1)
            
            # Normalize
            human_norm = F.normalize(human_flat, p=2, dim=1)
            ai_norm = F.normalize(ai_flat, p=2, dim=1)
            
            # Cosine similarity
            cosine_sim = (human_norm * ai_norm).sum(dim=1).mean()
            cca_loss = 1.0 - torch.clamp(cosine_sim, 0.0, 1.0)
        
        return cca_loss
    
    def compute_total_repgap_loss(self,
                                 human_reps: torch.Tensor,
                                 ai_reps: torch.Tensor,
                                 mapper: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute total representation gap loss
        
        Args:
            human_reps: [batch, human_dim] human representations
            ai_reps: [batch, ai_dim] AI representations
            mapper: Representation mapper T_ψ
            
        Returns:
            loss_dict: Dictionary containing loss components
        """
        # Map human representations
        mapped_human = mapper(human_reps)
        
        # Wasserstein distance loss (between mapped human and AI representations)
        wasserstein_loss = self.compute_wasserstein_loss(mapped_human, ai_reps)
        
        # CCA correlation loss (use differentiable version for training)
        cca_loss = self.compute_differentiable_cca_loss(mapped_human, ai_reps)
        
        # Total representation gap
        total_repgap_loss = self.mu * (wasserstein_loss + cca_loss)
        
        return {
            'repgap_loss': total_repgap_loss,
            'wasserstein_loss': wasserstein_loss,
            'cca_loss': cca_loss
        }


class SlicedWassersteinLoss:
    """
    Sliced Wasserstein distance for more efficient computation
    """
    
    def __init__(self, num_projections: int = 50):
        self.num_projections = num_projections
    
    def compute_sliced_wasserstein_loss(self,
                                      human_reps: torch.Tensor,
                                      mapped_reps: torch.Tensor) -> torch.Tensor:
        """
        Compute sliced Wasserstein distance
        
        Args:
            human_reps: [batch, dim] human representations
            mapped_reps: [batch, dim] mapped representations
            
        Returns:
            sliced_wasserstein_loss: Sliced Wasserstein distance
        """
        device = human_reps.device
        batch_size, dim = human_reps.shape
        
        # Generate random projections
        projections = torch.randn(self.num_projections, dim, device=device)
        projections = F.normalize(projections, p=2, dim=1)
        
        # Project data onto random directions
        human_projected = torch.mm(human_reps, projections.t())  # [batch, num_proj]
        mapped_projected = torch.mm(mapped_reps, projections.t())  # [batch, num_proj]
        
        # Compute 1D Wasserstein distances
        wasserstein_distances = []
        for i in range(self.num_projections):
            # Sort projections
            human_sorted, _ = torch.sort(human_projected[:, i])
            mapped_sorted, _ = torch.sort(mapped_projected[:, i])
            
            # 1D Wasserstein distance (L2)
            distance = torch.mean((human_sorted - mapped_sorted) ** 2)
            wasserstein_distances.append(distance)
        
        # Average over projections
        sliced_wasserstein = torch.stack(wasserstein_distances).mean()
        
        return sliced_wasserstein


def create_repgap_loss(config: Dict) -> RepresentationGapLoss:
    """Factory function to create representation gap loss"""
    return RepresentationGapLoss(
        mu=config.get('mu_rep', 0.1),
        cca_components=config.get('cca_components', 10),
        use_sinkhorn=config.get('use_sinkhorn', True),
        sinkhorn_reg=config.get('sinkhorn_reg', 0.1)
    )


def create_sliced_wasserstein_loss(config: Dict) -> SlicedWassersteinLoss:
    """Factory function to create sliced Wasserstein loss"""
    return SlicedWassersteinLoss(
        num_projections=config.get('num_projections', 50)
    )
