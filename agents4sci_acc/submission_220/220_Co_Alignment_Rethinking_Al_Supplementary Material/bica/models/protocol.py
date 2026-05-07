"""
Protocol Generator with Gumbel-Softmax for BiCA
Implements discrete protocol codes with IB/MDL regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


def gumbel_softmax_sample(logits: torch.Tensor, 
                         tau: float = 1.0, 
                         hard: bool = True) -> torch.Tensor:
    """
    Gumbel-Softmax sampling with straight-through estimator
    
    Args:
        logits: [batch, vocab_size] unnormalized logits
        tau: Temperature parameter
        hard: Whether to use straight-through estimator
        
    Returns:
        samples: [batch, vocab_size] one-hot (hard) or soft samples
    """
    # Sample Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    
    # Add noise and apply softmax with temperature
    y_soft = F.softmax((logits + gumbel_noise) / tau, dim=-1)
    
    if hard:
        # Straight-through estimator
        y_hard = torch.zeros_like(y_soft)
        y_hard.scatter_(-1, y_soft.argmax(-1, keepdim=True), 1.0)
        y = (y_hard - y_soft).detach() + y_soft
    else:
        y = y_soft
    
    return y


class ProtocolGenerator(nn.Module):
    """
    Protocol generator with Gumbel-Softmax for discrete codes.
    
    Architecture: MLP (ctx->128->code_dim), Gumbel-Softmax sampling
    Generates discrete protocol codes based on context (task state, uncertainty, errors)
    """
    
    def __init__(self,
                 context_dim: int = 64,
                 hidden_dim: int = 128,
                 code_dim: int = 16,
                 vocab_size: int = 64):
        super().__init__()
        
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.vocab_size = vocab_size
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Protocol code generator
        self.code_generator = nn.Linear(hidden_dim, code_dim)
        
        # Code to message mapping
        self.code_to_message = nn.Linear(code_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, context: torch.Tensor, 
                tau: float = 1.0, 
                hard: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            context: Context features [batch, context_dim]
            tau: Gumbel temperature
            hard: Use straight-through estimator
            
        Returns:
            message_logits: [batch, vocab_size] message logits
            code_probs: [batch, code_dim] protocol code probabilities
            code_samples: [batch, code_dim] sampled codes (one-hot or soft)
        """
        batch_size = context.size(0)
        
        # Encode context
        context_features = self.context_encoder(context)
        
        # Generate protocol codes
        code_logits = self.code_generator(context_features)
        
        # Sample codes using Gumbel-Softmax
        code_samples = gumbel_softmax_sample(code_logits, tau, hard)
        code_probs = F.softmax(code_logits, dim=-1)
        
        # Map codes to messages
        message_logits = self.code_to_message(code_samples)
        
        return message_logits, code_probs, code_samples
    
    def sample_message(self, context: torch.Tensor, 
                      tau: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample message from protocol"""
        message_logits, code_probs, code_samples = self.forward(context, tau, hard=True)
        
        # Sample message
        message_probs = F.softmax(message_logits, dim=-1)
        message = torch.multinomial(message_probs, 1).squeeze(-1)
        
        return message, code_probs
    
    def get_message_probs(self, context: torch.Tensor, 
                         tau: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get message probabilities"""
        message_logits, code_probs, _ = self.forward(context, tau, hard=False)
        message_probs = F.softmax(message_logits, dim=-1)
        return message_probs, code_probs
    
    def compute_ib_loss(self, code_probs: torch.Tensor, 
                       prior_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Information Bottleneck / MDL regularization loss
        
        Args:
            code_probs: [batch, code_dim] code probabilities
            prior_probs: [code_dim] prior over codes (uniform if None)
            
        Returns:
            ib_loss: Information bottleneck loss
        """
        batch_size = code_probs.size(0)
        
        if prior_probs is None:
            # Uniform prior
            prior_probs = torch.ones(self.code_dim, device=code_probs.device) / self.code_dim
        
        # Expand prior for batch
        prior_probs = prior_probs.unsqueeze(0).expand(batch_size, -1)
        
        # KL divergence from prior (encourages compact codes)
        kl_loss = F.kl_div(
            torch.log(prior_probs + 1e-8),
            code_probs,
            reduction='batchmean'
        )
        
        return kl_loss
    
    def compute_mutual_information(self, context: torch.Tensor, 
                                  code_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute mutual information I(C; M) between context and codes
        
        Args:
            context: [batch, context_dim] context features
            code_probs: [batch, code_dim] code probabilities
            
        Returns:
            mi_estimate: Mutual information estimate
        """
        # Simple MI estimate using marginal and conditional entropies
        # H(M) - H(M|C)
        
        # Marginal entropy H(M)
        marginal_code_probs = code_probs.mean(dim=0)  # [code_dim]
        h_marginal = -(marginal_code_probs * torch.log(marginal_code_probs + 1e-8)).sum()
        
        # Conditional entropy H(M|C)
        h_conditional = -(code_probs * torch.log(code_probs + 1e-8)).sum(dim=-1).mean()
        
        # MI estimate
        mi_estimate = h_marginal - h_conditional
        
        return mi_estimate


class ContextBuilder:
    """
    Helper class to build context features for protocol generator
    """
    
    def __init__(self, context_dim: int = 64):
        self.context_dim = context_dim
    
    def build_context(self, env_state: Dict, history: Dict, 
                     uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Build context vector from environment state and history
        
        Args:
            env_state: Environment state dictionary
            history: Interaction history
            uncertainty: Model uncertainty estimates
            
        Returns:
            context: [context_dim] context vector
        """
        context = np.zeros(self.context_dim, dtype=np.float32)
        
        # Environment features
        if 'agent_pos' in env_state:
            agent_pos = env_state['agent_pos']
            context[0:2] = agent_pos / 8.0  # Normalized position
        
        if 'goal_pos' in env_state:
            goal_pos = env_state['goal_pos']
            context[2:4] = goal_pos / 8.0  # Normalized goal
        
        if 'step_count' in env_state:
            context[4] = env_state['step_count'] / 60.0  # Normalized step count
        
        # Recent errors
        if 'recent_collisions' in history:
            context[5] = min(history['recent_collisions'], 5) / 5.0
        
        if 'recent_failures' in history:
            context[6] = min(history['recent_failures'], 3) / 3.0
        
        # Communication history features
        if 'message_history' in history and history['message_history']:
            recent_messages = history['message_history'][-5:]  # Last 5 messages
            for i, msg_data in enumerate(recent_messages):
                if i < 5:
                    base_idx = 7 + i * 3
                    if base_idx + 2 < self.context_dim:
                        context[base_idx] = msg_data.get('ai_message', 0) / 64.0
                        context[base_idx + 1] = msg_data.get('human_message', 0) / 32.0
                        context[base_idx + 2] = msg_data.get('instructor_action', 0) / 8.0
        
        # Uncertainty features
        if uncertainty is not None:
            uncertainty_np = uncertainty.detach().cpu().numpy()
            context[22:26] = uncertainty_np[:4] if len(uncertainty_np) >= 4 else [0, 0, 0, 0]
        
        # Task difficulty indicators
        if 'obstacle_density' in env_state:
            context[26] = env_state['obstacle_density']
        
        if 'distance_to_goal' in env_state:
            context[27] = min(env_state['distance_to_goal'], 16) / 16.0
        
        # Fill remaining with random features for now
        context[28:] = np.random.normal(0, 0.1, self.context_dim - 28)
        
        return torch.from_numpy(context).float()


def create_protocol_generator(config: Dict) -> ProtocolGenerator:
    """Factory function to create protocol generator"""
    return ProtocolGenerator(
        context_dim=config.get('context_dim', 64),
        hidden_dim=config.get('protocol_hidden_dim', 128),
        code_dim=config.get('code_dim', 16),
        vocab_size=config.get('ai_vocab_size', 64)
    )


def create_context_builder(config: Dict) -> ContextBuilder:
    """Factory function to create context builder"""
    return ContextBuilder(
        context_dim=config.get('context_dim', 64)
    )
