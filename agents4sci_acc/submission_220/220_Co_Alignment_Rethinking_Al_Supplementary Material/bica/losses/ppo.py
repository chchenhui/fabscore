"""
PPO Loss Implementation with KL-to-Prior for BiCA
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


def compute_gae(rewards: torch.Tensor,
                values: torch.Tensor,
                dones: torch.Tensor,
                gamma: float = 0.99,
                lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: [batch, seq_len] rewards
        values: [batch, seq_len] value estimates
        dones: [batch, seq_len] done flags
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        advantages: [batch, seq_len] advantage estimates
        returns: [batch, seq_len] discounted returns
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    
    # Compute advantages using GAE
    gae = 0
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = 0  # Terminal state
        else:
            next_value = values[:, t + 1]
        
        # TD error
        delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values[:, t]
        
        # GAE computation
        gae = delta + gamma * lam * (1 - dones[:, t]) * gae
        advantages[:, t] = gae
    
    # Compute returns
    returns = advantages + values
    
    return advantages, returns


class PPOLoss:
    """
    PPO Loss with KL-to-Prior regularization for BiCA
    """
    
    def __init__(self,
                 clip_epsilon: float = 0.2,
                 value_coeff: float = 0.5,
                 entropy_coeff: float = 0.01,
                 kl_coeff: float = 0.02,
                 max_grad_norm: float = 1.0):
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.kl_coeff = kl_coeff
        self.max_grad_norm = max_grad_norm
    
    def compute_policy_loss(self,
                           new_log_probs: torch.Tensor,
                           old_log_probs: torch.Tensor,
                           advantages: torch.Tensor,
                           prior_log_probs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute PPO policy loss with KL-to-prior
        
        Args:
            new_log_probs: [batch] log probs from current policy
            old_log_probs: [batch] log probs from old policy (detached)
            advantages: [batch] advantage estimates
            prior_log_probs: [batch] log probs from prior policy (optional)
            
        Returns:
            loss_dict: Dictionary containing loss components
        """
        # PPO ratio
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        
        # Surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        
        # PPO clipped loss (negative because we want to maximize)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL to prior (if provided)
        kl_prior_loss = 0.0
        if prior_log_probs is not None:
            kl_prior_loss = self.kl_coeff * (new_log_probs - prior_log_probs.detach()).mean()
        
        # Total policy loss
        total_policy_loss = policy_loss + kl_prior_loss
        
        # Compute statistics
        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean()
            clipped_fraction = ((ratio > 1 + self.clip_epsilon) | 
                              (ratio < 1 - self.clip_epsilon)).float().mean()
        
        return {
            'policy_loss': total_policy_loss,
            'ppo_loss': policy_loss,
            'kl_prior_loss': kl_prior_loss,
            'approx_kl': approx_kl,
            'clipped_fraction': clipped_fraction
        }
    
    def compute_value_loss(self,
                          new_values: torch.Tensor,
                          old_values: torch.Tensor,
                          returns: torch.Tensor,
                          clip_values: bool = True) -> torch.Tensor:
        """
        Compute value function loss
        
        Args:
            new_values: [batch] current value estimates
            old_values: [batch] old value estimates
            returns: [batch] target returns
            clip_values: Whether to clip value updates
            
        Returns:
            value_loss: Value function loss
        """
        if clip_values:
            # Clipped value loss (similar to policy clipping)
            value_pred_clipped = old_values + torch.clamp(
                new_values - old_values, -self.clip_epsilon, self.clip_epsilon
            )
            
            value_losses = (new_values - returns) ** 2
            value_losses_clipped = (value_pred_clipped - returns) ** 2
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            # Simple MSE loss
            value_loss = F.mse_loss(new_values, returns)
        
        return value_loss
    
    def compute_entropy_loss(self, entropy: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy bonus
        
        Args:
            entropy: [batch] policy entropy
            
        Returns:
            entropy_loss: Negative entropy (to be minimized)
        """
        return -entropy.mean()  # Negative because we want to maximize entropy
    
    def compute_total_loss(self,
                          policy_outputs: Dict[str, torch.Tensor],
                          value_outputs: Dict[str, torch.Tensor],
                          batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute total PPO loss
        
        Args:
            policy_outputs: Policy model outputs
            value_outputs: Value model outputs  
            batch_data: Batch training data
            
        Returns:
            loss_dict: Complete loss breakdown
        """
        # Extract required tensors
        new_log_probs = policy_outputs['log_probs']
        old_log_probs = batch_data['old_log_probs']
        advantages = batch_data['advantages']
        returns = batch_data['returns']
        
        new_values = value_outputs['values'].squeeze(-1)
        old_values = batch_data['old_values']
        
        # Optional prior log probs
        prior_log_probs = batch_data.get('prior_log_probs', None)
        
        # Compute policy loss
        policy_loss_dict = self.compute_policy_loss(
            new_log_probs, old_log_probs, advantages, prior_log_probs
        )
        
        # Compute value loss
        value_loss = self.compute_value_loss(new_values, old_values, returns)
        
        # Compute entropy loss
        entropy_loss = self.compute_entropy_loss(policy_outputs['entropy'])
        
        # Total loss
        total_loss = (policy_loss_dict['policy_loss'] + 
                     self.value_coeff * value_loss +
                     self.entropy_coeff * entropy_loss)
        
        # Combine all losses
        loss_dict = {
            'total_loss': total_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            **policy_loss_dict
        }
        
        return loss_dict


def kl_categorical(p_probs: torch.Tensor, q_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between categorical distributions
    
    Args:
        p_probs: [batch, vocab] probabilities of distribution P
        q_probs: [batch, vocab] probabilities of distribution Q
        
    Returns:
        kl_div: [batch] KL divergence D_KL(P||Q)
    """
    # Add small epsilon for numerical stability
    eps = 1e-8
    p_probs = p_probs + eps
    q_probs = q_probs + eps
    
    # Compute KL divergence
    kl_div = (p_probs * torch.log(p_probs / q_probs)).sum(dim=-1)
    
    return kl_div


class DualUpdater:
    """
    Dual variable updates for KL budget constraints
    """
    
    def __init__(self, 
                 alpha_lambda: float = 0.01,
                 tau_a: float = 0.05,
                 tau_h: float = 0.03):
        self.alpha_lambda = alpha_lambda
        self.tau_a = tau_a  # AI KL budget
        self.tau_h = tau_h  # Human KL budget
        
        # Dual variables
        self.lambda_a = 0.02
        self.lambda_h = 0.01
    
    def update_dual_variables(self, 
                            observed_kl_a: float,
                            observed_kl_h: float) -> Dict[str, float]:
        """
        Update dual variables based on observed KL divergences
        
        Args:
            observed_kl_a: Observed AI KL divergence from prior
            observed_kl_h: Observed human KL divergence from prior
            
        Returns:
            dual_info: Dictionary with dual variable info
        """
        # Update AI dual variable
        self.lambda_a = max(0.0, self.lambda_a + self.alpha_lambda * (observed_kl_a - self.tau_a))
        
        # Update human dual variable
        self.lambda_h = max(0.0, self.lambda_h + self.alpha_lambda * (observed_kl_h - self.tau_h))
        
        return {
            'lambda_a': self.lambda_a,
            'lambda_h': self.lambda_h,
            'kl_a_violation': observed_kl_a - self.tau_a,
            'kl_h_violation': observed_kl_h - self.tau_h
        }
    
    def get_kl_coefficients(self) -> Tuple[float, float]:
        """Get current KL regularization coefficients"""
        return self.lambda_a, self.lambda_h


def create_ppo_loss(config: Dict) -> PPOLoss:
    """Factory function to create PPO loss"""
    return PPOLoss(
        clip_epsilon=config.get('ppo_clip', 0.2),
        value_coeff=config.get('value_coeff', 0.5),
        entropy_coeff=config.get('entropy_coeff', 0.01),
        kl_coeff=config.get('kl_coeff', 0.02),
        max_grad_norm=config.get('max_grad_norm', 1.0)
    )


def create_dual_updater(config: Dict) -> DualUpdater:
    """Factory function to create dual updater"""
    return DualUpdater(
        alpha_lambda=config.get('alpha_lambda', 0.01),
        tau_a=config.get('kl_budget_a', 0.05),
        tau_h=config.get('kl_budget_h', 0.03)
    )
