"""
AI Policy and Value Network implementations for BiCA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union
import numpy as np


class AIPolicy(nn.Module):
    """
    AI agent policy network.
    
    Architecture: MLP (obs->256->256, Tanh), optional GRU(128), categorical head
    Takes observations and human messages as input, outputs action probabilities.
    """
    
    def __init__(self, 
                 obs_dim: int = 18,  # 3x3x2 patch + 4 heading
                 message_dim: int = 32,
                 hidden_dim: int = 256,
                 gru_hidden: int = 128,
                 action_dim: int = 4,
                 use_gru: bool = True):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.message_dim = message_dim
        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden
        self.action_dim = action_dim
        self.use_gru = use_gru
        
        # Message embedding
        self.message_embed = nn.Embedding(32, message_dim)  # Human message vocab
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim + message_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Optional GRU for temporal modeling
        if use_gru:
            self.gru = nn.GRU(hidden_dim, gru_hidden, batch_first=True)
            policy_input_dim = gru_hidden
        else:
            self.gru = None
            policy_input_dim = hidden_dim
        
        # Policy head
        self.policy_head = nn.Linear(policy_input_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
    
    def forward(self, obs: torch.Tensor, human_message: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            obs: Agent observations [batch, obs_dim]
            human_message: Human messages [batch] (indices)
            hidden: GRU hidden state [batch, gru_hidden]
            
        Returns:
            action_logits: [batch, action_dim]
            new_hidden: [batch, gru_hidden] or None
        """
        batch_size = obs.size(0)
        
        # Embed human message
        msg_embed = self.message_embed(human_message)  # [batch, message_dim]
        
        # Concatenate obs and message
        input_features = torch.cat([obs, msg_embed], dim=-1)
        
        # Encode observations
        encoded = self.obs_encoder(input_features)  # [batch, hidden_dim]
        
        # Apply GRU if used
        if self.use_gru:
            if hidden is None:
                hidden = torch.zeros(1, batch_size, self.gru_hidden, 
                                   device=obs.device, dtype=obs.dtype)
            
            # GRU expects [batch, seq_len, features]
            gru_input = encoded.unsqueeze(1)  # [batch, 1, hidden_dim]
            gru_output, new_hidden = self.gru(gru_input, hidden)
            features = gru_output.squeeze(1)  # [batch, gru_hidden]
        else:
            features = encoded
            new_hidden = None
        
        # Get action logits
        action_logits = self.policy_head(features)
        
        return action_logits, new_hidden
    
    def get_action_probs(self, obs: torch.Tensor, human_message: torch.Tensor,
                        hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action probabilities"""
        logits, new_hidden = self.forward(obs, human_message, hidden)
        probs = F.softmax(logits, dim=-1)
        return probs, new_hidden
    
    def sample_action(self, obs: torch.Tensor, human_message: torch.Tensor,
                     hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        probs, new_hidden = self.get_action_probs(obs, human_message, hidden)
        action = torch.multinomial(probs, 1).squeeze(-1)
        log_prob = torch.log(probs.gather(1, action.unsqueeze(-1))).squeeze(-1)
        return action, log_prob, new_hidden
    
    def log_prob(self, obs: torch.Tensor, human_message: torch.Tensor, 
                 actions: torch.Tensor, hidden: Optional[torch.Tensor] = None,
                 return_entropy: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute log probability of given actions"""
        probs, _ = self.get_action_probs(obs, human_message, hidden)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(-1))).squeeze(-1)
        
        if return_entropy:
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            return log_probs, entropy
            
        return log_probs
    
    def entropy(self, obs: torch.Tensor, human_message: torch.Tensor,
               hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute policy entropy"""
        probs, _ = self.get_action_probs(obs, human_message, hidden)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return entropy


class ValueNet(nn.Module):
    """
    Value network for advantage estimation (GAE)
    """
    
    def __init__(self, 
                 obs_dim: int = 18,
                 message_dim: int = 32,
                 hidden_dim: int = 256,
                 gru_hidden: int = 128,
                 use_gru: bool = True):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.message_dim = message_dim
        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden
        self.use_gru = use_gru
        
        # Message embedding
        self.message_embed = nn.Embedding(32, message_dim)
        
        # Feature encoder (shared with policy)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + message_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Optional GRU
        if use_gru:
            self.gru = nn.GRU(hidden_dim, gru_hidden, batch_first=True)
            value_input_dim = gru_hidden
        else:
            self.gru = None
            value_input_dim = hidden_dim
        
        # Value head
        self.value_head = nn.Linear(value_input_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
    
    def forward(self, obs: torch.Tensor, human_message: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            obs: Agent observations [batch, obs_dim]
            human_message: Human messages [batch] (indices)
            hidden: GRU hidden state [batch, gru_hidden]
            
        Returns:
            values: State values [batch, 1]
            new_hidden: [batch, gru_hidden] or None
        """
        batch_size = obs.size(0)
        
        # Embed human message
        msg_embed = self.message_embed(human_message)
        
        # Concatenate obs and message
        input_features = torch.cat([obs, msg_embed], dim=-1)
        
        # Encode features
        encoded = self.encoder(input_features)
        
        # Apply GRU if used
        if self.use_gru:
            if hidden is None:
                hidden = torch.zeros(1, batch_size, self.gru_hidden,
                                   device=obs.device, dtype=obs.dtype)
            
            gru_input = encoded.unsqueeze(1)
            gru_output, new_hidden = self.gru(gru_input, hidden)
            features = gru_output.squeeze(1)
        else:
            features = encoded
            new_hidden = None
        
        # Get value
        values = self.value_head(features)
        
        return values, new_hidden


def preprocess_ai_observation(ai_obs: np.ndarray, ai_heading: np.ndarray) -> torch.Tensor:
    """
    Preprocess AI observation for network input
    
    Args:
        ai_obs: [3, 3, 2] egocentric patch
        ai_heading: [4] one-hot heading
        
    Returns:
        obs_tensor: [18] flattened observation
    """
    obs_flat = ai_obs.flatten()  # 18 dims
    heading_flat = ai_heading  # 4 dims
    combined = np.concatenate([obs_flat, heading_flat])  # 22 dims total
    return torch.from_numpy(combined).float()


def create_ai_policy(config: Dict) -> AIPolicy:
    """Factory function to create AI policy"""
    return AIPolicy(
        obs_dim=config.get('ai_obs_dim', 22),  # 3*3*2 + 4
        message_dim=config.get('message_embed_dim', 32),
        hidden_dim=config.get('policy_hidden_dim', 256),
        gru_hidden=config.get('gru_hidden_dim', 128),
        action_dim=config.get('action_dim', 4),
        use_gru=config.get('use_gru', True)
    )


def create_value_net(config: Dict) -> ValueNet:
    """Factory function to create value network"""
    return ValueNet(
        obs_dim=config.get('ai_obs_dim', 22),
        message_dim=config.get('message_embed_dim', 32),
        hidden_dim=config.get('value_hidden_dim', 256),
        gru_hidden=config.get('gru_hidden_dim', 128),
        use_gru=config.get('use_gru', True)
    )
