"""
Human Surrogate Model for BiCA
Implements GRU-based human surrogate with protocol table state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class ProtocolTable:
    """
    Internal protocol table for human surrogate.
    Maps semantic meanings to token sequences.
    """
    
    def __init__(self, vocab_size: int = 32):
        self.vocab_size = vocab_size
        self.table = {}
        self.update_prob = 0.1  # Probability of updating protocol
        
        # Initialize with basic mappings
        self._init_basic_protocol()
    
    def _init_basic_protocol(self):
        """Initialize basic protocol mappings"""
        # Directions
        self.table['north'] = [1, 2]  # N, 2
        self.table['east'] = [3, 2]   # E, 2  
        self.table['south'] = [4, 2]  # S, 2
        self.table['west'] = [5, 2]   # W, 2
        
        # Counts
        self.table['one'] = [6]
        self.table['two'] = [7]
        self.table['three'] = [8]
        self.table['four'] = [9]
        
        # Landmarks/macros
        self.table['junction'] = [10]  # J
        self.table['deadend'] = [11]   # D
        self.table['turn_around'] = [12, 13]  # TURN-A
        self.table['align'] = [14]     # ALIGN
        
        # Special tokens
        self.table['<pad>'] = [0]
        self.table['<eos>'] = [15]
    
    def get_tokens(self, semantic: str, add_noise: bool = False, 
                   noise_prob: float = 0.05) -> List[int]:
        """Get tokens for semantic meaning"""
        if semantic not in self.table:
            return [0]  # Padding if unknown
        
        tokens = self.table[semantic].copy()
        
        # Add noise if requested
        if add_noise:
            for i in range(len(tokens)):
                if np.random.random() < noise_prob:
                    tokens[i] = np.random.randint(1, self.vocab_size)
        
        return tokens
    
    def update_mapping(self, semantic: str, tokens: List[int]):
        """Update protocol mapping"""
        if np.random.random() < self.update_prob:
            self.table[semantic] = tokens
    
    def get_state_vector(self) -> np.ndarray:
        """Get protocol table as state vector for neural network"""
        # Simple encoding: concatenate all mappings
        state = np.zeros(64, dtype=np.float32)  # Fixed size encoding
        
        idx = 0
        for key, tokens in self.table.items():
            if idx + len(tokens) < 64:
                state[idx:idx+len(tokens)] = tokens
                idx += len(tokens)
            if idx >= 60:  # Leave some space
                break
        
        return state


class HumanSurrogate(nn.Module):
    """
    Human surrogate model with GRU and protocol table state.
    
    Architecture: GRU(128) over [o^H_t, Emb(m^A_t), Emb(u_t)] -> categorical m^H_t
    Maintains internal protocol table state.
    """
    
    def __init__(self,
                 human_obs_dim: int = 192,  # 8*8*3 full map
                 ai_message_dim: int = 32,
                 instructor_dim: int = 16,
                 protocol_state_dim: int = 64,
                 gru_hidden: int = 128,
                 human_vocab_size: int = 32):
        super().__init__()
        
        self.human_obs_dim = human_obs_dim
        self.ai_message_dim = ai_message_dim
        self.instructor_dim = instructor_dim
        self.protocol_state_dim = protocol_state_dim
        self.gru_hidden = gru_hidden
        self.human_vocab_size = human_vocab_size
        
        # Protocol table (not part of neural network)
        self.protocol_table = ProtocolTable(human_vocab_size)
        
        # Embedding layers
        self.ai_message_embed = nn.Embedding(64, ai_message_dim)  # AI message vocab
        self.instructor_embed = nn.Embedding(8, instructor_dim)   # Instructor actions
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(human_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Protocol state encoder
        self.protocol_encoder = nn.Sequential(
            nn.Linear(protocol_state_dim, 32),
            nn.ReLU()
        )
        
        # Input dimension for GRU
        gru_input_dim = 64 + ai_message_dim + instructor_dim + 32  # obs + ai_msg + instr + protocol
        
        # Main GRU
        self.gru = nn.GRU(gru_input_dim, gru_hidden, batch_first=True)
        
        # Output head for human messages
        self.message_head = nn.Linear(gru_hidden, human_vocab_size)
        
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
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.1)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize GRU hidden state"""
        return torch.zeros(1, batch_size, self.gru_hidden, device=device)
    
    def forward(self, 
                human_obs: torch.Tensor,
                ai_message: torch.Tensor,
                instructor_action: torch.Tensor,
                hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            human_obs: Human observations [batch, human_obs_dim]
            ai_message: AI messages [batch] (indices)
            instructor_action: Instructor actions [batch] (indices)
            hidden: GRU hidden state [1, batch, gru_hidden]
            
        Returns:
            message_logits: [batch, human_vocab_size]
            new_hidden: [1, batch, gru_hidden]
        """
        batch_size = human_obs.size(0)
        device = human_obs.device
        
        # Encode human observations
        obs_encoded = self.obs_encoder(human_obs)  # [batch, 64]
        
        # Embed AI message and instructor action
        ai_msg_embed = self.ai_message_embed(ai_message)  # [batch, ai_message_dim]
        instr_embed = self.instructor_embed(instructor_action)  # [batch, instructor_dim]
        
        # Get protocol state (batch-wise for now, could be individualized)
        protocol_state = torch.from_numpy(
            np.stack([self.protocol_table.get_state_vector()] * batch_size)
        ).float().to(device)
        protocol_encoded = self.protocol_encoder(protocol_state)  # [batch, 32]
        
        # Concatenate all inputs
        gru_input = torch.cat([
            obs_encoded,
            ai_msg_embed,
            instr_embed,
            protocol_encoded
        ], dim=-1)  # [batch, gru_input_dim]
        
        # Add sequence dimension for GRU
        gru_input = gru_input.unsqueeze(1)  # [batch, 1, gru_input_dim]
        
        # GRU forward pass
        gru_output, new_hidden = self.gru(gru_input, hidden)
        gru_output = gru_output.squeeze(1)  # [batch, gru_hidden]
        
        # Get message logits
        message_logits = self.message_head(gru_output)
        
        return message_logits, new_hidden
    
    def sample_message(self,
                      human_obs: torch.Tensor,
                      ai_message: torch.Tensor,
                      instructor_action: torch.Tensor,
                      hidden: torch.Tensor,
                      temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample human message"""
        logits, new_hidden = self.forward(human_obs, ai_message, instructor_action, hidden)
        
        # Apply temperature
        logits = logits / temperature
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        message = torch.multinomial(probs, 1).squeeze(-1)
        log_prob = F.log_softmax(logits, dim=-1).gather(1, message.unsqueeze(-1)).squeeze(-1)
        
        return message, log_prob, new_hidden
    
    def get_message_probs(self,
                         human_obs: torch.Tensor,
                         ai_message: torch.Tensor,
                         instructor_action: torch.Tensor,
                         hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get message probabilities"""
        logits, new_hidden = self.forward(human_obs, ai_message, instructor_action, hidden)
        probs = F.softmax(logits, dim=-1)
        return probs, new_hidden
    
    def log_prob(self,
                 human_obs: torch.Tensor,
                 ai_message: torch.Tensor,
                 instructor_action: torch.Tensor,
                 hidden: torch.Tensor,
                 target_messages: torch.Tensor) -> torch.Tensor:
        """Compute log probability of target messages"""
        logits, _ = self.forward(human_obs, ai_message, instructor_action, hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(1, target_messages.unsqueeze(-1)).squeeze(-1)
    
    def update_protocol(self, ai_message: int, instructor_action: int):
        """Update protocol table based on AI message and instructor action"""
        # Simple update rule - could be made more sophisticated
        if instructor_action > 0:  # If instructor intervened
            # This is where protocol adaptation would happen
            # For now, just a placeholder
            pass
    
    def add_noise(self, enable: bool = True, noise_prob: float = 0.05):
        """Enable/disable noise in protocol table"""
        self.protocol_table.update_prob = noise_prob if enable else 0.0
    
    def set_ood_mode(self, ood: bool = True):
        """Set out-of-distribution mode (higher noise)"""
        if ood:
            self.add_noise(True, 0.1)
        else:
            self.add_noise(True, 0.05)


def preprocess_human_observation(human_obs: np.ndarray) -> torch.Tensor:
    """
    Preprocess human observation for network input
    
    Args:
        human_obs: [8, 8, 3] full map observation
        
    Returns:
        obs_tensor: [192] flattened observation
    """
    obs_flat = human_obs.flatten()
    return torch.from_numpy(obs_flat).float()


def create_human_surrogate(config: Dict) -> HumanSurrogate:
    """Factory function to create human surrogate"""
    return HumanSurrogate(
        human_obs_dim=config.get('human_obs_dim', 192),  # 8*8*3
        ai_message_dim=config.get('ai_message_embed_dim', 32),
        instructor_dim=config.get('instructor_embed_dim', 16),
        protocol_state_dim=config.get('protocol_state_dim', 64),
        gru_hidden=config.get('human_gru_hidden', 128),
        human_vocab_size=config.get('human_vocab_size', 32)
    )
