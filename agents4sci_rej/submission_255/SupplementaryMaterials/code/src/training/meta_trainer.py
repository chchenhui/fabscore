"""
Training Pipeline for Hierarchical Meta-Learning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
import time
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import logging
import wandb
from pathlib import Path

from ..models.hierarchical_maml import HierarchicalMAML, MetaLearner
from ..data.preprocessing import MetaLearningDataLoader


class HierarchicalMetaTrainer:
    """
    Main training class for hierarchical meta-learning.
    """
    
    def __init__(self,
                 model: HierarchicalMAML,
                 train_loader: MetaLearningDataLoader,
                 val_loader: MetaLearningDataLoader,
                 meta_lr: float = 0.001,
                 inner_lr: float = 0.01,
                 inner_steps: int = 5,
                 hierarchy_weights: List[float] = [1.0, 0.7, 0.5],
                 device: str = 'cuda',
                 save_dir: str = './checkpoints',
                 log_dir: str = './logs',
                 use_wandb: bool = True,
                 wandb_project: str = 'hierarchical-meta-learning'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.hierarchy_weights = hierarchy_weights
        
        # Initialize meta-learner
        self.meta_learner = MetaLearner(
            model=model,
            inner_lr=inner_lr,
            meta_lr=meta_lr,
            inner_steps=inner_steps
        )
        
        # Setup directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(self.log_dir)
        
        # Setup wandb
        if use_wandb:
            wandb.init(
                project=wandb_project,
                config={
                    'meta_lr': meta_lr,
                    'inner_lr': inner_lr,
                    'inner_steps': inner_steps,
                    'hierarchy_weights': hierarchy_weights,
                    'n_way': train_loader.n_way,
                    'k_shot': train_loader.k_shot,
                    'n_query': train_loader.n_query
                }
            )
            wandb.watch(model)
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.training_history = {
            'train_loss': [],
            'val_accuracy': [],
            'val_loss': []
        }
        
    def train(self, 
              num_epochs: int = 100,
              tasks_per_epoch: int = 1000,
              val_frequency: int = 10,
              save_frequency: int = 20,
              early_stopping_patience: int = 20) -> Dict:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of training epochs
            tasks_per_epoch: Number of meta-learning tasks per epoch
            val_frequency: How often to run validation
            save_frequency: How often to save checkpoints
            early_stopping_patience: Patience for early stopping
            
        Returns:
            training_results: Dictionary with training history and final metrics
        """
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(tasks_per_epoch)
            
            # Validation phase
            if epoch % val_frequency == 0:
                val_metrics = self._validate_epoch()
                
                # Check for improvement
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    patience_counter += 1
                
                # Log metrics
                self._log_metrics(train_metrics, val_metrics, epoch)
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Regular checkpoint saving
            if epoch % save_frequency == 0:
                self._save_checkpoint(is_best=False)
        
        # Final evaluation
        final_metrics = self._final_evaluation()
        
        # Cleanup
        self.writer.close()
        if wandb.run:
            wandb.finish()
        
        return {
            'training_history': self.training_history,
            'final_metrics': final_metrics,
            'best_val_accuracy': self.best_val_acc
        }
    
    def _train_epoch(self, tasks_per_epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_components = {'organ': 0.0, 'histology': 0.0, 'molecular': 0.0}
        
        # Progress bar
        pbar = tqdm(range(tasks_per_epoch), desc=f"Epoch {self.current_epoch}")
        
        for task_idx in pbar:
            # Get batch of tasks
            task_batch = self.train_loader.get_batch()
            
            # Meta-update
            meta_loss_info = self.meta_learner.meta_update(task_batch)
            
            epoch_loss += meta_loss_info['meta_loss']
            for level, loss_val in meta_loss_info['loss_components'].items():
                epoch_components[level] += loss_val
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{meta_loss_info['meta_loss']:.4f}",
                'mol': f"{meta_loss_info['loss_components']['molecular']:.3f}"
            })
        
        # Average metrics
        avg_loss = epoch_loss / tasks_per_epoch
        avg_components = {level: loss / tasks_per_epoch for level, loss in epoch_components.items()}
        
        # Store in history
        self.training_history['train_loss'].append(avg_loss)
        
        return {
            'loss': avg_loss,
            'loss_components': avg_components
        }
    
    def _validate_epoch(self) -> Dict:
        """Validate for one epoch."""
        self.model.eval()
        
        total_tasks = 100  # Number of validation tasks
        total_accuracy = 0.0
        total_loss = 0.0
        level_accuracies = {'organ': 0.0, 'histology': 0.0, 'molecular': 0.0}
        
        with torch.no_grad():
            for _ in range(total_tasks):
                # Get validation task
                task = self.val_loader.sample_task()
                support_x, support_y = task['support']
                query_x, query_y = task['query']
                
                # Move to device
                support_x = support_x.to(self.device)
                query_x = query_x.to(self.device)
                for level in support_y:
                    support_y[level] = support_y[level].to(self.device)
                    query_y[level] = query_y[level].to(self.device)
                
                # Fast adaptation
                adapted_model = self.meta_learner.fast_adapt(support_x, support_y)
                adapted_model.eval()
                
                # Evaluate on query set
                query_predictions, _ = adapted_model(query_x)
                
                # Compute accuracies for each level
                for level in ['organ', 'histology', 'molecular']:
                    pred_labels = query_predictions[level].argmax(dim=1)
                    accuracy = (pred_labels == query_y[level]).float().mean()
                    level_accuracies[level] += accuracy.item()
                
                # Primary accuracy (molecular level)
                molecular_acc = (query_predictions['molecular'].argmax(dim=1) == query_y['molecular']).float().mean()
                total_accuracy += molecular_acc.item()
                
                # Compute loss
                from ..models.hierarchical_maml import hierarchical_loss
                val_loss, _ = hierarchical_loss(query_predictions, query_y, self.hierarchy_weights)
                total_loss += val_loss.item()
        
        # Average metrics
        avg_accuracy = total_accuracy / total_tasks
        avg_loss = total_loss / total_tasks
        avg_level_accuracies = {level: acc / total_tasks for level, acc in level_accuracies.items()}
        
        # Store in history
        self.training_history['val_accuracy'].append(avg_accuracy)
        self.training_history['val_loss'].append(avg_loss)
        
        return {
            'accuracy': avg_accuracy,
            'loss': avg_loss,
            'level_accuracies': avg_level_accuracies
        }
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics to tensorboard and wandb."""
        # Tensorboard logging
        self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
        self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
        self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
        
        for level, loss in train_metrics['loss_components'].items():
            self.writer.add_scalar(f'Train/Loss_{level}', loss, epoch)
        
        for level, acc in val_metrics['level_accuracies'].items():
            self.writer.add_scalar(f'Val/Accuracy_{level}', acc, epoch)
        
        # Wandb logging
        if wandb.run:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/loss': val_metrics['loss']
            }
            
            for level, loss in train_metrics['loss_components'].items():
                log_dict[f'train/loss_{level}'] = loss
            
            for level, acc in val_metrics['level_accuracies'].items():
                log_dict[f'val/accuracy_{level}'] = acc
            
            wandb.log(log_dict)
        
        # Console logging
        self.logger.info(
            f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
            f"Val Acc={val_metrics['accuracy']:.4f}, Val Loss={val_metrics['loss']:.4f}"
        )
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.meta_learner.meta_optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'training_history': self.training_history,
            'hierarchy_weights': self.hierarchy_weights
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with accuracy: {self.best_val_acc:.4f}")
    
    def _final_evaluation(self) -> Dict:
        """Perform final comprehensive evaluation."""
        self.logger.info("Performing final evaluation...")
        
        # Load best model
        best_model_path = self.save_dir / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        # Evaluate on different shot scenarios
        shot_scenarios = [1, 5, 10]
        final_results = {}
        
        for k_shot in shot_scenarios:
            # Create temporary loader with different shot setting
            temp_loader = MetaLearningDataLoader(
                pathway_data=self.val_loader.pathway_data,
                cancer_types=self.val_loader.cancer_types,
                hierarchical_labels=self.val_loader.hierarchical_labels,
                n_way=self.val_loader.n_way,
                k_shot=k_shot,
                n_query=self.val_loader.n_query,
                n_tasks_per_batch=1
            )
            
            # Evaluate
            accuracies = []
            for _ in range(100):  # 100 tasks for robust evaluation
                task = temp_loader.sample_task()
                support_x, support_y = task['support']
                query_x, query_y = task['query']
                
                # Move to device
                support_x = support_x.to(self.device)
                query_x = query_x.to(self.device)
                for level in support_y:
                    support_y[level] = support_y[level].to(self.device)
                    query_y[level] = query_y[level].to(self.device)
                
                # Fast adaptation
                adapted_model = self.meta_learner.fast_adapt(support_x, support_y)
                adapted_model.eval()
                
                # Evaluate
                with torch.no_grad():
                    query_predictions, _ = adapted_model(query_x)
                    accuracy = (query_predictions['molecular'].argmax(dim=1) == query_y['molecular']).float().mean()
                    accuracies.append(accuracy.item())
            
            final_results[f'{k_shot}_shot_accuracy'] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'ci_95': 1.96 * np.std(accuracies) / np.sqrt(len(accuracies))
            }
        
        return final_results


def create_model_and_trainer(data_splits: Dict, 
                           config: Dict,
                           device: str = 'cuda') -> Tuple[HierarchicalMAML, HierarchicalMetaTrainer]:
    """
    Factory function to create model and trainer.
    
    Args:
        data_splits: Preprocessed data splits
        config: Configuration dictionary
        device: Computing device
        
    Returns:
        model: Initialized hierarchical MAML model
        trainer: Initialized trainer
    """
    # Create model
    model = HierarchicalMAML(
        input_dim=data_splits['train']['pathway_data'].shape[1],
        hidden_dims=config.get('hidden_dims', [64, 128, 64]),
        feature_dim=config.get('feature_dim', 32),
        num_organ_classes=len(np.unique(data_splits['train']['organ_labels'])),
        num_histology_classes=len(np.unique(data_splits['train']['histology_labels'])),
        num_molecular_classes=len(np.unique(data_splits['train']['molecular_labels'])),
        use_attention=config.get('use_attention', True),
        dropout_rate=config.get('dropout_rate', 0.1)
    )
    
    # Create data loaders
    train_loader = MetaLearningDataLoader(
        pathway_data=data_splits['train']['pathway_data'],
        cancer_types=data_splits['train']['cancer_types'],
        hierarchical_labels={
            'organ': data_splits['train']['organ_labels'],
            'histology': data_splits['train']['histology_labels'],
            'molecular': data_splits['train']['molecular_labels']
        },
        n_way=config.get('n_way', 5),
        k_shot=config.get('k_shot', 5),
        n_query=config.get('n_query', 15),
        n_tasks_per_batch=config.get('n_tasks_per_batch', 8)
    )
    
    val_loader = MetaLearningDataLoader(
        pathway_data=data_splits['val']['pathway_data'],
        cancer_types=data_splits['val']['cancer_types'],
        hierarchical_labels={
            'organ': data_splits['val']['organ_labels'],
            'histology': data_splits['val']['histology_labels'],
            'molecular': data_splits['val']['molecular_labels']
        },
        n_way=config.get('n_way', 5),
        k_shot=config.get('k_shot', 5),
        n_query=config.get('n_query', 15),
        n_tasks_per_batch=1  # Single task for validation
    )
    
    # Create trainer
    trainer = HierarchicalMetaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        meta_lr=config.get('meta_lr', 0.001),
        inner_lr=config.get('inner_lr', 0.01),
        inner_steps=config.get('inner_steps', 5),
        hierarchy_weights=config.get('hierarchy_weights', [1.0, 0.7, 0.5]),
        device=device,
        save_dir=config.get('save_dir', './checkpoints'),
        log_dir=config.get('log_dir', './logs'),
        use_wandb=config.get('use_wandb', True)
    )
    
    return model, trainer