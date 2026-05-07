"""
Information Bottleneck and MDL Loss for BiCA Protocol Learning
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class IBLoss:
    """
    Information Bottleneck loss for protocol learning
    
    Implements I_φ(M;C) ≈ E[D_KL(p_φ(m|c) || p(m))]
    Encourages compact yet expressive protocol messages
    """
    
    def __init__(self, beta: float = 1.0):
        self.beta = beta
    
    def compute_ib_loss(self,
                       message_probs: torch.Tensor,
                       context: torch.Tensor,
                       prior_probs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute Information Bottleneck loss
        
        Args:
            message_probs: [batch, vocab_size] message probabilities p_φ(m|c)
            context: [batch, context_dim] context vectors
            prior_probs: [vocab_size] prior message probabilities p(m)
            
        Returns:
            loss_dict: Dictionary containing IB loss components
        """
        batch_size, vocab_size = message_probs.shape
        
        # Default uniform prior if not provided
        if prior_probs is None:
            prior_probs = torch.ones(vocab_size, device=message_probs.device) / vocab_size
        
        # Expand prior for batch computation
        prior_expanded = prior_probs.unsqueeze(0).expand(batch_size, -1)
        
        # KL divergence from prior for each context
        kl_divergences = F.kl_div(
            torch.log(prior_expanded + 1e-8),
            message_probs,
            reduction='none'
        ).sum(dim=-1)  # [batch]
        
        # Average over batch (expectation over contexts)
        ib_loss = kl_divergences.mean()
        
        # Compute mutual information estimate
        mi_estimate = self._estimate_mutual_information(message_probs, context)
        
        return {
            'ib_loss': self.beta * ib_loss,
            'kl_from_prior': ib_loss,
            'mutual_information': mi_estimate
        }
    
    def _estimate_mutual_information(self,
                                   message_probs: torch.Tensor,
                                   context: torch.Tensor) -> torch.Tensor:
        """
        Estimate mutual information I(M; C) between messages and context
        
        Args:
            message_probs: [batch, vocab_size] message probabilities
            context: [batch, context_dim] context vectors
            
        Returns:
            mi_estimate: Mutual information estimate
        """
        # Simple MI estimate using entropy differences
        # I(M; C) = H(M) - H(M|C)
        
        # Marginal message distribution
        marginal_probs = message_probs.mean(dim=0)  # [vocab_size]
        
        # Marginal entropy H(M)
        h_marginal = -(marginal_probs * torch.log(marginal_probs + 1e-8)).sum()
        
        # Conditional entropy H(M|C) (average over contexts)
        h_conditional = -(message_probs * torch.log(message_probs + 1e-8)).sum(dim=-1).mean()
        
        # MI estimate
        mi_estimate = h_marginal - h_conditional
        
        return mi_estimate
    
    def compute_compression_loss(self, code_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute compression loss to encourage sparse code usage
        
        Args:
            code_probs: [batch, code_dim] protocol code probabilities
            
        Returns:
            compression_loss: Loss encouraging sparsity
        """
        # Encourage sparsity in code usage
        # Use negative entropy to encourage peaky distributions
        entropy = -(code_probs * torch.log(code_probs + 1e-8)).sum(dim=-1).mean()
        compression_loss = entropy  # Minimize entropy = maximize sparsity
        
        return compression_loss


class MDLLoss:
    """
    Minimum Description Length (MDL) loss for protocol learning
    
    Implements a more principled approach to protocol compression
    based on coding theory principles
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 vocab_size: int = 64):
        self.alpha = alpha
        self.vocab_size = vocab_size
    
    def compute_mdl_loss(self,
                        message_probs: torch.Tensor,
                        code_probs: torch.Tensor,
                        context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute MDL loss for protocol learning
        
        Args:
            message_probs: [batch, vocab_size] message probabilities
            code_probs: [batch, code_dim] protocol code probabilities
            context: [batch, context_dim] context vectors
            
        Returns:
            loss_dict: Dictionary containing MDL loss components
        """
        # Description length of messages given codes
        message_description_length = self._compute_message_description_length(message_probs)
        
        # Description length of codes (model complexity)
        code_description_length = self._compute_code_description_length(code_probs)
        
        # Total MDL loss
        mdl_loss = message_description_length + self.alpha * code_description_length
        
        return {
            'mdl_loss': mdl_loss,
            'message_dl': message_description_length,
            'code_dl': code_description_length
        }
    
    def _compute_message_description_length(self, message_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute description length of messages
        
        Args:
            message_probs: [batch, vocab_size] message probabilities
            
        Returns:
            description_length: Average description length in bits
        """
        # Negative log probability gives description length in nats
        # Convert to bits by dividing by log(2)
        log_probs = torch.log(message_probs + 1e-8)
        description_length = -log_probs.sum(dim=-1).mean() / np.log(2)
        
        return description_length
    
    def _compute_code_description_length(self, code_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute description length of protocol codes (model complexity)
        
        Args:
            code_probs: [batch, code_dim] protocol code probabilities
            
        Returns:
            code_complexity: Code complexity penalty
        """
        # Use entropy as a measure of code complexity
        # Higher entropy = more complex code distribution
        entropy = -(code_probs * torch.log(code_probs + 1e-8)).sum(dim=-1).mean()
        
        # Convert to bits
        code_complexity = entropy / np.log(2)
        
        return code_complexity
    
    def compute_adaptive_mdl_loss(self,
                                 message_probs: torch.Tensor,
                                 code_probs: torch.Tensor,
                                 context: torch.Tensor,
                                 task_performance: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive MDL loss that balances compression with task performance
        
        Args:
            message_probs: [batch, vocab_size] message probabilities
            code_probs: [batch, code_dim] protocol code probabilities
            context: [batch, context_dim] context vectors
            task_performance: [batch] task performance scores
            
        Returns:
            loss_dict: Dictionary containing adaptive MDL loss components
        """
        # Base MDL loss
        base_mdl = self.compute_mdl_loss(message_probs, code_probs, context)
        
        # Adaptive weighting based on task performance
        # Lower performance = less compression pressure
        performance_weight = torch.sigmoid(task_performance - 0.5)  # [batch]
        adaptive_weight = performance_weight.mean()
        
        # Weighted MDL loss
        adaptive_mdl_loss = adaptive_weight * base_mdl['mdl_loss']
        
        return {
            'adaptive_mdl_loss': adaptive_mdl_loss,
            'performance_weight': adaptive_weight,
            **base_mdl
        }


class ProtocolRegularizer:
    """
    Combined regularization for protocol learning
    
    Combines IB and MDL approaches with additional regularization terms
    """
    
    def __init__(self,
                 ib_weight: float = 1.0,
                 mdl_weight: float = 0.5,
                 diversity_weight: float = 0.1,
                 consistency_weight: float = 0.2):
        self.ib_loss = IBLoss(beta=ib_weight)
        self.mdl_loss = MDLLoss(alpha=mdl_weight)
        self.diversity_weight = diversity_weight
        self.consistency_weight = consistency_weight
    
    def compute_total_regularization(self,
                                   message_probs: torch.Tensor,
                                   code_probs: torch.Tensor,
                                   context: torch.Tensor,
                                   prior_message_probs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total protocol regularization loss
        
        Args:
            message_probs: [batch, vocab_size] message probabilities
            code_probs: [batch, code_dim] protocol code probabilities
            context: [batch, context_dim] context vectors
            prior_message_probs: [vocab_size] prior message distribution
            
        Returns:
            loss_dict: Complete regularization loss breakdown
        """
        # IB loss
        ib_dict = self.ib_loss.compute_ib_loss(message_probs, context, prior_message_probs)
        
        # MDL loss
        mdl_dict = self.mdl_loss.compute_mdl_loss(message_probs, code_probs, context)
        
        # Diversity loss (encourage diverse message usage)
        diversity_loss = self._compute_diversity_loss(message_probs)
        
        # Consistency loss (similar contexts should produce similar messages)
        consistency_loss = self._compute_consistency_loss(message_probs, context)
        
        # Total regularization
        total_reg_loss = (ib_dict['ib_loss'] + 
                         mdl_dict['mdl_loss'] +
                         self.diversity_weight * diversity_loss +
                         self.consistency_weight * consistency_loss)
        
        # Combine all components
        loss_dict = {
            'total_regularization': total_reg_loss,
            'diversity_loss': diversity_loss,
            'consistency_loss': consistency_loss,
            **ib_dict,
            **mdl_dict
        }
        
        return loss_dict
    
    def _compute_diversity_loss(self, message_probs: torch.Tensor) -> torch.Tensor:
        """Encourage diverse message usage across batch"""
        # Marginal distribution over messages
        marginal = message_probs.mean(dim=0)
        
        # Maximize entropy of marginal (encourage diversity)
        entropy = -(marginal * torch.log(marginal + 1e-8)).sum()
        diversity_loss = -entropy  # Negative because we want to maximize
        
        return diversity_loss
    
    def _compute_consistency_loss(self, 
                                message_probs: torch.Tensor, 
                                context: torch.Tensor) -> torch.Tensor:
        """Encourage consistent messages for similar contexts"""
        batch_size = context.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=context.device)
        
        # Compute pairwise context similarities
        context_norm = F.normalize(context, p=2, dim=-1)
        context_sim = torch.mm(context_norm, context_norm.t())  # [batch, batch]
        
        # Compute pairwise message similarities
        message_sim = torch.mm(message_probs, message_probs.t())  # [batch, batch]
        
        # Consistency loss: high context similarity should lead to high message similarity
        consistency_loss = F.mse_loss(message_sim, context_sim)
        
        return consistency_loss


def create_ib_loss(config: Dict) -> IBLoss:
    """Factory function to create IB loss"""
    return IBLoss(beta=config.get('beta_ib', 1.0))


def create_mdl_loss(config: Dict) -> MDLLoss:
    """Factory function to create MDL loss"""
    return MDLLoss(
        alpha=config.get('alpha_mdl', 1.0),
        vocab_size=config.get('ai_vocab_size', 64)
    )


def create_protocol_regularizer(config: Dict) -> ProtocolRegularizer:
    """Factory function to create protocol regularizer"""
    return ProtocolRegularizer(
        ib_weight=config.get('ib_weight', 1.0),
        mdl_weight=config.get('mdl_weight', 0.5),
        diversity_weight=config.get('diversity_weight', 0.1),
        consistency_weight=config.get('consistency_weight', 0.2)
    )
