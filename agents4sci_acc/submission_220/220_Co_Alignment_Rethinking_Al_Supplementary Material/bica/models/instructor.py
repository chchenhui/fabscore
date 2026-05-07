"""
Instructor Model π_I for BiCA
Provides teaching interventions based on uncertainty and errors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque


class InstructorModel(nn.Module):
    """
    Instructor model π_I^ω(u_t | history)
    
    Provides discrete interventions based on:
    - Model uncertainty
    - Recent errors
    - Out-of-distribution flags
    - Communication breakdowns
    """
    
    def __init__(self,
                 history_dim: int = 64,
                 hidden_dim: int = 128,
                 gru_hidden: int = 64,
                 num_interventions: int = 8):
        super().__init__()
        
        self.history_dim = history_dim
        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden
        self.num_interventions = num_interventions
        
        # History encoder
        self.history_encoder = nn.Sequential(
            nn.Linear(history_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # GRU for temporal modeling
        self.gru = nn.GRU(hidden_dim, gru_hidden, batch_first=True)
        
        # Intervention head
        self.intervention_head = nn.Linear(gru_hidden, num_interventions)
        
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
    
    def forward(self, history_features: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            history_features: [batch, seq_len, history_dim] history features
            hidden: [1, batch, gru_hidden] GRU hidden state
            
        Returns:
            intervention_logits: [batch, num_interventions]
            new_hidden: [1, batch, gru_hidden]
        """
        batch_size, seq_len = history_features.shape[:2]
        
        # Encode history
        history_encoded = self.history_encoder(history_features)  # [batch, seq_len, hidden_dim]
        
        # GRU forward pass
        gru_output, new_hidden = self.gru(history_encoded, hidden)
        
        # Use last output for intervention prediction
        last_output = gru_output[:, -1, :]  # [batch, gru_hidden]
        
        # Get intervention logits
        intervention_logits = self.intervention_head(last_output)
        
        return intervention_logits, new_hidden
    
    def sample_intervention(self, history_features: torch.Tensor,
                          hidden: Optional[torch.Tensor] = None,
                          temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample intervention action"""
        logits, new_hidden = self.forward(history_features, hidden)
        
        # Apply temperature
        logits = logits / temperature
        
        # Sample intervention
        probs = F.softmax(logits, dim=-1)
        intervention = torch.multinomial(probs, 1).squeeze(-1)
        log_prob = F.log_softmax(logits, dim=-1).gather(1, intervention.unsqueeze(-1)).squeeze(-1)
        
        return intervention, log_prob, new_hidden
    
    def get_intervention_probs(self, history_features: torch.Tensor,
                             hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get intervention probabilities"""
        logits, new_hidden = self.forward(history_features, hidden)
        probs = F.softmax(logits, dim=-1)
        return probs, new_hidden


class HistoryFeatureExtractor:
    """
    Extracts features from interaction history for instructor model
    """
    
    def __init__(self, history_dim: int = 64, window_size: int = 10):
        self.history_dim = history_dim
        self.window_size = window_size
        
        # Track various signals
        self.error_history = deque(maxlen=window_size)
        self.uncertainty_history = deque(maxlen=window_size)
        self.communication_history = deque(maxlen=window_size)
        self.performance_history = deque(maxlen=window_size)
    
    def update_history(self, 
                      step_info: Dict,
                      uncertainty: Optional[float] = None,
                      communication_success: Optional[bool] = None):
        """Update history with new step information"""
        
        # Error indicators
        error_signal = 0.0
        if step_info.get('collision', False):
            error_signal += 1.0
        if step_info.get('timeout', False):
            error_signal += 0.5
        if step_info.get('wrong_action', False):
            error_signal += 0.3
        
        self.error_history.append(error_signal)
        
        # Uncertainty
        if uncertainty is not None:
            self.uncertainty_history.append(uncertainty)
        else:
            self.uncertainty_history.append(0.0)
        
        # Communication success
        comm_signal = 1.0 if communication_success else 0.0
        self.communication_history.append(comm_signal)
        
        # Performance (reward-based)
        reward = step_info.get('reward', 0.0)
        normalized_reward = max(0.0, (reward + 10) / 60.0)  # Normalize to [0, 1]
        self.performance_history.append(normalized_reward)
    
    def extract_features(self, current_state: Dict) -> np.ndarray:
        """
        Extract history features for instructor model
        
        Args:
            current_state: Current environment/model state
            
        Returns:
            features: [history_dim] feature vector
        """
        features = np.zeros(self.history_dim, dtype=np.float32)
        
        # Recent error rate
        if self.error_history:
            features[0] = np.mean(self.error_history)
            features[1] = np.max(self.error_history)
            features[2] = len([x for x in self.error_history if x > 0.5]) / len(self.error_history)
        
        # Uncertainty trends
        if self.uncertainty_history:
            features[3] = np.mean(self.uncertainty_history)
            features[4] = np.std(self.uncertainty_history)
            features[5] = self.uncertainty_history[-1] if self.uncertainty_history else 0.0
        
        # Communication effectiveness
        if self.communication_history:
            features[6] = np.mean(self.communication_history)
            # Get last 3 items safely
            recent_items = list(self.communication_history)[-3:] if len(self.communication_history) >= 3 else list(self.communication_history)
            features[7] = np.sum(recent_items) / max(len(recent_items), 1)  # Recent success
        
        # Performance trends
        if self.performance_history:
            features[8] = np.mean(self.performance_history)
            if len(self.performance_history) > 1:
                features[9] = self.performance_history[-1] - self.performance_history[-2]  # Trend
        
        # Current state features
        if 'step_count' in current_state:
            features[10] = current_state['step_count'] / 60.0  # Normalized step count
        
        if 'distance_to_goal' in current_state:
            features[11] = min(current_state['distance_to_goal'], 16) / 16.0
        
        if 'model_confidence' in current_state:
            features[12] = current_state['model_confidence']
        
        if 'ood_detected' in current_state:
            features[13] = 1.0 if current_state['ood_detected'] else 0.0
        
        # Message complexity features
        if 'recent_message_lengths' in current_state:
            msg_lengths = current_state['recent_message_lengths']
            if msg_lengths:
                features[14] = np.mean(msg_lengths) / 10.0  # Normalized
                features[15] = np.std(msg_lengths) / 5.0
        
        # Repetition detection
        if len(self.communication_history) >= 3:
            recent_comm = list(self.communication_history)[-3:]
            if len(set(recent_comm)) == 1:  # All same
                features[16] = 1.0  # High repetition
        
        # Time since last intervention
        if 'steps_since_intervention' in current_state:
            features[17] = min(current_state['steps_since_intervention'], 20) / 20.0
        
        # Fill remaining with derived features
        for i in range(18, min(32, self.history_dim)):
            if i < len(features):
                # Simple combinations of existing features
                features[i] = features[i % 18] * features[(i + 1) % 18]
        
        # Pad or truncate to exact dimension
        if len(features) > self.history_dim:
            features = features[:self.history_dim]
        elif len(features) < self.history_dim:
            padding = np.zeros(self.history_dim - len(features))
            features = np.concatenate([features, padding])
        
        return features


class InterventionExecutor:
    """
    Executes instructor interventions
    """
    
    INTERVENTIONS = {
        0: "no_action",
        1: "show_legend",
        2: "highlight_error", 
        3: "provide_example",
        4: "suggest_protocol",
        5: "reset_protocol",
        6: "show_goal",
        7: "encourage"
    }
    
    def __init__(self):
        self.intervention_costs = {
            0: 0.0,    # no_action
            1: 0.1,    # show_legend
            2: 0.05,   # highlight_error
            3: 0.15,   # provide_example
            4: 0.2,    # suggest_protocol
            5: 0.25,   # reset_protocol
            6: 0.05,   # show_goal
            7: 0.02    # encourage
        }
        
        self.intervention_history = deque(maxlen=50)
    
    def execute_intervention(self, intervention_id: int, 
                           context: Dict) -> Dict:
        """
        Execute an intervention and return its effects
        
        Args:
            intervention_id: ID of intervention to execute
            context: Current context (environment, models, etc.)
            
        Returns:
            intervention_effects: Dictionary describing effects
        """
        intervention_name = self.INTERVENTIONS.get(intervention_id, "unknown")
        cost = self.intervention_costs.get(intervention_id, 0.1)
        
        # Record intervention
        self.intervention_history.append({
            'id': intervention_id,
            'name': intervention_name,
            'cost': cost,
            'step': context.get('step_count', 0)
        })
        
        effects = {
            'intervention_id': intervention_id,
            'intervention_name': intervention_name,
            'cost': cost,
            'executed': True
        }
        
        # Specific intervention effects
        if intervention_id == 1:  # show_legend
            effects['protocol_clarity'] = 0.2
            
        elif intervention_id == 2:  # highlight_error
            effects['error_awareness'] = 0.3
            
        elif intervention_id == 3:  # provide_example
            effects['learning_boost'] = 0.25
            
        elif intervention_id == 4:  # suggest_protocol
            effects['protocol_update'] = True
            effects['protocol_clarity'] = 0.3
            
        elif intervention_id == 5:  # reset_protocol
            effects['protocol_reset'] = True
            effects['protocol_clarity'] = 0.5
            
        elif intervention_id == 6:  # show_goal
            effects['goal_clarity'] = 0.4
            
        elif intervention_id == 7:  # encourage
            effects['motivation_boost'] = 0.1
        
        return effects
    
    def get_total_cost(self) -> float:
        """Get total intervention cost"""
        return sum(intervention['cost'] for intervention in self.intervention_history)
    
    def get_recent_interventions(self, window: int = 5) -> List[Dict]:
        """Get recent interventions"""
        return list(self.intervention_history)[-window:]


def create_instructor(config: Dict) -> InstructorModel:
    """Factory function to create instructor model"""
    return InstructorModel(
        history_dim=config.get('history_dim', 64),
        hidden_dim=config.get('instructor_hidden_dim', 128),
        gru_hidden=config.get('instructor_gru_hidden', 64),
        num_interventions=config.get('num_interventions', 8)
    )


def create_history_extractor(config: Dict) -> HistoryFeatureExtractor:
    """Factory function to create history feature extractor"""
    return HistoryFeatureExtractor(
        history_dim=config.get('history_dim', 64),
        window_size=config.get('history_window_size', 10)
    )


def create_intervention_executor() -> InterventionExecutor:
    """Factory function to create intervention executor"""
    return InterventionExecutor()
