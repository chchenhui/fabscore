"""
Hierarchical Model-Agnostic Meta-Learning (MAML) Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher
from typing import Dict, List, Tuple, Optional
from .pathway_encoder import PathwayEncoder, HierarchicalAttention


class HierarchicalMAML(nn.Module):
    """
    Hierarchical MAML model for cancer pathway signature classification.
    Implements 3-level hierarchy: organ system -> histology -> molecular subtype.
    """
    
    def __init__(self,
                 input_dim: int = 32,
                 hidden_dims: List[int] = [64, 128, 64],
                 feature_dim: int = 32,
                 num_organ_classes: int = 9,
                 num_histology_classes: int = 4,
                 num_molecular_classes: int = 36,
                 use_attention: bool = True,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        # Pathway encoder
        self.encoder = PathwayEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=feature_dim,
            dropout_rate=dropout_rate,
            use_pathway_attention=use_attention
        )
        
        # Hierarchical attention mechanism
        if use_attention:
            self.hierarchical_attention = HierarchicalAttention(
                feature_dim=feature_dim,
                num_levels=3
            )
        else:
            self.hierarchical_attention = None
        
        # Hierarchical classifiers
        self.level_classifiers = nn.ModuleDict({
            'organ': nn.Linear(feature_dim, num_organ_classes),
            'histology': nn.Linear(feature_dim, num_histology_classes),
            'molecular': nn.Linear(feature_dim, num_molecular_classes)
        })
        
        # Hierarchical attention weights (learnable)
        self.level_weights = nn.Parameter(torch.ones(3))
        
        # Store class numbers for reference
        self.num_classes = {
            'organ': num_organ_classes,
            'histology': num_histology_classes,
            'molecular': num_molecular_classes
        }
        
    def forward(self, 
                x: torch.Tensor, 
                adapted_params: Optional[Dict] = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass through hierarchical model.
        
        Args:
            x: Input pathway signatures [batch_size, 32]
            adapted_params: Adapted parameters from meta-learning
            
        Returns:
            predictions: Dictionary of predictions for each level
            features: Encoded features
        """
        # Encode pathway signatures
        if adapted_params is not None and 'encoder' in adapted_params:
            # Use adapted encoder parameters
            features = self._forward_encoder_with_params(x, adapted_params['encoder'])
        else:
            features, _ = self.encoder(x)
        
        # Hierarchical predictions
        predictions = {}
        
        for level in ['organ', 'histology', 'molecular']:
            # Apply hierarchical attention if available
            if self.hierarchical_attention is not None:
                attended_features, _ = self.hierarchical_attention(features, level)
            else:
                attended_features = features
            
            # Classification for this level
            if adapted_params is not None and level in adapted_params:
                # Use adapted classifier parameters
                pred = F.linear(
                    attended_features,
                    adapted_params[level]['weight'],
                    adapted_params[level]['bias']
                )
            else:
                pred = self.level_classifiers[level](attended_features)
            
            predictions[level] = pred
        
        return predictions, features
    
    def _forward_encoder_with_params(self, x: torch.Tensor, encoder_params: Dict) -> torch.Tensor:
        """Forward pass through encoder with adapted parameters."""
        # This is a simplified version - in practice, you'd need to handle
        # the full encoder architecture with adapted parameters
        encoded = x
        
        # Apply encoding layers with adapted parameters
        for i, layer in enumerate(self.encoder.layers):
            if isinstance(layer, nn.Linear):
                weight_key = f'layers.{i}.weight'
                bias_key = f'layers.{i}.bias'
                if weight_key in encoder_params and bias_key in encoder_params:
                    encoded = F.linear(encoded, encoder_params[weight_key], encoder_params[bias_key])
                else:
                    encoded = layer(encoded)
            else:
                encoded = layer(encoded)
        
        return encoded


def hierarchical_loss(predictions: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor],
                     hierarchy_weights: List[float] = [1.0, 0.7, 0.5],
                     label_smoothing: float = 0.0) -> torch.Tensor:
    """
    Compute hierarchical loss combining all levels.
    
    Args:
        predictions: Dictionary of predictions for each level
        targets: Dictionary of targets for each level
        hierarchy_weights: Weights for each hierarchy level
        label_smoothing: Label smoothing factor
        
    Returns:
        total_loss: Combined hierarchical loss
    """
    total_loss = 0.0
    loss_components = {}
    
    levels = ['organ', 'histology', 'molecular']
    
    for i, level in enumerate(levels):
        if level in predictions and level in targets:
            if label_smoothing > 0:
                level_loss = label_smoothed_cross_entropy(
                    predictions[level], targets[level], label_smoothing
                )
            else:
                level_loss = F.cross_entropy(predictions[level], targets[level])
            
            weighted_loss = hierarchy_weights[i] * level_loss
            total_loss += weighted_loss
            loss_components[level] = level_loss.item()
    
    return total_loss, loss_components


def label_smoothed_cross_entropy(predictions: torch.Tensor,
                                targets: torch.Tensor,
                                smoothing: float = 0.1) -> torch.Tensor:
    """Apply label smoothing to cross-entropy loss."""
    num_classes = predictions.size(-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(predictions)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    
    return torch.mean(torch.sum(-true_dist * F.log_softmax(predictions, dim=1), dim=1))


class MetaLearner:
    """
    Meta-learner for hierarchical MAML training.
    """
    
    def __init__(self,
                 model: HierarchicalMAML,
                 inner_lr: float = 0.01,
                 meta_lr: float = 0.001,
                 inner_steps: int = 5,
                 first_order: bool = False):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        
    def meta_update(self, tasks: List[Dict]) -> Dict:
        """
        Perform meta-update using batch of tasks.
        
        Args:
            tasks: List of meta-learning tasks
            
        Returns:
            meta_loss_info: Information about meta-loss and components
        """
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        meta_loss_components = {'organ': 0.0, 'histology': 0.0, 'molecular': 0.0}
        
        for task in tasks:
            task_loss, task_components = self._single_task_update(task)
            meta_loss += task_loss
            
            for level, loss_val in task_components.items():
                meta_loss_components[level] += loss_val
        
        # Average over tasks
        meta_loss /= len(tasks)
        for level in meta_loss_components:
            meta_loss_components[level] /= len(tasks)
        
        # Meta-gradient step
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.meta_optimizer.step()
        
        return {
            'meta_loss': meta_loss.item(),
            'loss_components': meta_loss_components
        }
    
    def _single_task_update(self, task: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Perform inner loop adaptation for a single task.
        
        Args:
            task: Single meta-learning task
            
        Returns:
            query_loss: Loss on query set after adaptation
            loss_components: Loss components for each level
        """
        support_x, support_y = task['support']
        query_x, query_y = task['query']
        
        # Inner loop adaptation using higher library
        with higher.innerloop_ctx(
            self.model, 
            self.meta_optimizer,
            copy_initial_weights=False,
            track_higher_grads=not self.first_order
        ) as (fmodel, diffopt):
            
            # Inner loop updates
            for step in range(self.inner_steps):
                support_predictions, _ = fmodel(support_x)
                support_loss, _ = hierarchical_loss(support_predictions, support_y)
                
                diffopt.step(support_loss)
            
            # Evaluate on query set
            query_predictions, _ = fmodel(query_x)
            query_loss, loss_components = hierarchical_loss(query_predictions, query_y)
        
        return query_loss, loss_components
    
    def fast_adapt(self, 
                   support_x: torch.Tensor, 
                   support_y: Dict[str, torch.Tensor],
                   num_steps: Optional[int] = None) -> nn.Module:
        """
        Fast adaptation for few-shot evaluation.
        
        Args:
            support_x: Support set inputs
            support_y: Support set targets
            num_steps: Number of adaptation steps (default: self.inner_steps)
            
        Returns:
            adapted_model: Model adapted to support set
        """
        if num_steps is None:
            num_steps = self.inner_steps
        
        # Clone model for adaptation
        adapted_model = type(self.model)(
            input_dim=self.model.encoder.input_dim,
            feature_dim=self.model.encoder.output_dim,
            num_organ_classes=self.model.num_classes['organ'],
            num_histology_classes=self.model.num_classes['histology'],
            num_molecular_classes=self.model.num_classes['molecular']
        )
        adapted_model.load_state_dict(self.model.state_dict())
        adapted_model.eval()
        
        # Create optimizer for adaptation
        adapt_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Adaptation steps
        for step in range(num_steps):
            adapt_optimizer.zero_grad()
            
            predictions, _ = adapted_model(support_x)
            loss, _ = hierarchical_loss(predictions, support_y)
            
            loss.backward()
            adapt_optimizer.step()
        
        return adapted_model


def create_hierarchy_mapping() -> Dict:
    """
    Create mapping from cancer types to hierarchical labels.
    
    Returns:
        hierarchy_map: Mapping from cancer type to hierarchy levels
    """
    # Organ system mapping (Level 1)
    organ_mapping = {
        # Gastrointestinal
        'COAD': 0, 'READ': 0, 'STAD': 0, 'ESCA': 0, 'LIHC': 0, 'PAAD': 0, 'CHOL': 0,
        # Genitourinary  
        'KIRC': 1, 'KIRP': 1, 'KICH': 1, 'BLCA': 1, 'PRAD': 1, 'TGCT': 1, 
        'CESC': 1, 'UCEC': 1, 'OV': 1,
        # Thoracic
        'LUAD': 2, 'LUSC': 2, 'MESO': 2, 'THYM': 2,
        # Hematologic
        'LAML': 3, 'DLBC': 3, 'THCA': 3,
        # Nervous System
        'GBM': 4, 'LGG': 4,
        # Skin/Soft Tissue
        'SKCM': 5, 'SARC': 5, 'UCS': 5,
        # Head/Neck
        'HNSC': 6,
        # Breast
        'BRCA': 7,
        # Other
        'ACC': 8, 'PCPG': 8, 'UVM': 8
    }
    
    # Histology mapping (Level 2)
    histology_mapping = {
        # Adenocarcinoma
        'COAD': 0, 'READ': 0, 'STAD': 0, 'LUAD': 0, 'PAAD': 0, 
        'PRAD': 0, 'BRCA': 0, 'UCEC': 0, 'OV': 0, 'THCA': 0,
        # Squamous Cell Carcinoma
        'ESCA': 1, 'LUSC': 1, 'HNSC': 1, 'CESC': 1, 'BLCA': 1,
        # Sarcoma
        'SARC': 2, 'UCS': 2, 'MESO': 2,
        # Hematologic/Other
        'LAML': 3, 'DLBC': 3, 'GBM': 3, 'LGG': 3, 'SKCM': 3,
        'KIRC': 3, 'KIRP': 3, 'KICH': 3, 'LIHC': 3, 'CHOL': 3,
        'TGCT': 3, 'THYM': 3, 'ACC': 3, 'PCPG': 3, 'UVM': 3
    }
    
    # Molecular subtype mapping (Level 3) - direct cancer type indices
    molecular_mapping = {cancer_type: i for i, cancer_type in enumerate([
        'ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
        'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV',
        'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM',
        'UCEC', 'UCS', 'UVM'
    ])}
    
    # Add missing cancer types with appropriate mappings
    for cancer_type in ['COADREAD', 'GBMLGG']:
        if cancer_type not in molecular_mapping:
            molecular_mapping[cancer_type] = len(molecular_mapping)
    
    return {
        'organ': organ_mapping,
        'histology': histology_mapping,
        'molecular': molecular_mapping,
        'organ_names': [
            'Gastrointestinal', 'Genitourinary', 'Thoracic', 'Hematologic',
            'Nervous System', 'Skin/Soft Tissue', 'Head/Neck', 'Breast', 'Other'
        ],
        'histology_names': [
            'Adenocarcinoma', 'Squamous Cell Carcinoma', 'Sarcoma', 'Other'
        ]
    }