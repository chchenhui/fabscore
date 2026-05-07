"""
Baseline Models for Comparison with Hierarchical Meta-Learning
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import logging


class StandardNeuralNetwork(nn.Module):
    """
    Standard feedforward neural network for comparison.
    """
    
    def __init__(self, 
                 input_dim: int = 32,
                 hidden_dims: List[int] = [64, 128, 64],
                 num_classes: int = 36,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [num_classes]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if i < len(dims) - 2:  # Not the last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                layers.append(nn.BatchNorm1d(dims[i + 1]))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class HierarchicalNeuralNetwork(nn.Module):
    """
    Hierarchical neural network without meta-learning.
    """
    
    def __init__(self,
                 input_dim: int = 32,
                 hidden_dims: List[int] = [64, 128, 64],
                 feature_dim: int = 32,
                 num_organ_classes: int = 9,
                 num_histology_classes: int = 4,
                 num_molecular_classes: int = 36,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        # Feature extractor
        layers = []
        dims = [input_dim] + hidden_dims + [feature_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Hierarchical classifiers
        self.organ_classifier = nn.Linear(feature_dim, num_organ_classes)
        self.histology_classifier = nn.Linear(feature_dim, num_histology_classes)
        self.molecular_classifier = nn.Linear(feature_dim, num_molecular_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        return {
            'organ': self.organ_classifier(features),
            'histology': self.histology_classifier(features),
            'molecular': self.molecular_classifier(features)
        }


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for few-shot learning comparison.
    """
    
    def __init__(self,
                 input_dim: int = 32,
                 hidden_dims: List[int] = [64, 128, 64],
                 embedding_dim: int = 64):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [embedding_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, support_x, support_y, query_x):
        """
        Forward pass for prototypical networks.
        
        Args:
            support_x: Support set inputs [n_support, input_dim]
            support_y: Support set labels [n_support]
            query_x: Query set inputs [n_query, input_dim]
            
        Returns:
            logits: Classification logits for query set
        """
        # Encode support and query sets
        support_embeddings = self.encoder(support_x)
        query_embeddings = self.encoder(query_x)
        
        # Compute prototypes (class centroids)
        n_classes = len(torch.unique(support_y))
        prototypes = []
        
        for c in range(n_classes):
            class_mask = (support_y == c)
            class_embeddings = support_embeddings[class_mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        
        # Compute distances and logits
        distances = torch.cdist(query_embeddings, prototypes)
        logits = -distances  # Negative distance as logits
        
        return logits


class BaselineComparator:
    """
    Class for comprehensive baseline comparisons.
    """
    
    def __init__(self, data_splits: Dict, device: str = 'cuda'):
        self.data_splits = data_splits
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Extract data for sklearn models
        self.X_train = data_splits['train']['pathway_data']
        self.y_train = data_splits['train']['molecular_labels']
        self.X_val = data_splits['val']['pathway_data']
        self.y_val = data_splits['val']['molecular_labels']
        self.X_test = data_splits['test']['pathway_data']
        self.y_test = data_splits['test']['molecular_labels']
        
        # Results storage
        self.results = {}
        
    def evaluate_sklearn_baselines(self) -> Dict:
        """Evaluate traditional machine learning baselines."""
        self.logger.info("Evaluating sklearn baselines...")
        
        baselines = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            ),
            'SVM_RBF': SVC(
                kernel='rbf', 
                random_state=42, 
                probability=True
            ),
            'SVM_Linear': SVC(
                kernel='linear', 
                random_state=42, 
                probability=True
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, 
                max_iter=1000
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=42, 
                eval_metric='mlogloss',
                verbosity=0
            ),
            'LightGBM': lgb.LGBMClassifier(
                random_state=42, 
                verbosity=-1
            )
        }
        
        sklearn_results = {}
        
        for name, model in baselines.items():
            self.logger.info(f"Training {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            train_pred = model.predict(self.X_train)
            val_pred = model.predict(self.X_val)
            test_pred = model.predict(self.X_test)
            
            # Accuracies
            train_acc = accuracy_score(self.y_train, train_pred)
            val_acc = accuracy_score(self.y_val, val_pred)
            test_acc = accuracy_score(self.y_test, test_pred)
            
            sklearn_results[name] = {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'model': model
            }
            
            self.logger.info(f"{name} - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        
        self.results['sklearn'] = sklearn_results
        return sklearn_results
    
    def evaluate_neural_networks(self, epochs: int = 100) -> Dict:
        """Evaluate neural network baselines."""
        self.logger.info("Evaluating neural network baselines...")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(self.X_train).to(self.device)
        y_train_tensor = torch.LongTensor(self.y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(self.X_val).to(self.device)
        y_val_tensor = torch.LongTensor(self.y_val).to(self.device)
        X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
        y_test_tensor = torch.LongTensor(self.y_test).to(self.device)
        
        nn_results = {}
        
        # Standard Neural Network
        std_nn = StandardNeuralNetwork(
            input_dim=self.X_train.shape[1],
            num_classes=len(np.unique(self.y_train))
        ).to(self.device)
        
        std_nn_results = self._train_neural_network(
            std_nn, X_train_tensor, y_train_tensor, 
            X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor,
            epochs=epochs, model_name="StandardNN"
        )
        nn_results['StandardNN'] = std_nn_results
        
        # Hierarchical Neural Network (without meta-learning)
        hier_nn = HierarchicalNeuralNetwork(
            input_dim=self.X_train.shape[1],
            num_organ_classes=len(np.unique(self.data_splits['train']['organ_labels'])),
            num_histology_classes=len(np.unique(self.data_splits['train']['histology_labels'])),
            num_molecular_classes=len(np.unique(self.y_train))
        ).to(self.device)
        
        hier_nn_results = self._train_hierarchical_nn(
            hier_nn, epochs=epochs
        )
        nn_results['HierarchicalNN'] = hier_nn_results
        
        self.results['neural_networks'] = nn_results
        return nn_results
    
    def _train_neural_network(self, 
                            model: nn.Module,
                            X_train: torch.Tensor,
                            y_train: torch.Tensor,
                            X_val: torch.Tensor,
                            y_val: torch.Tensor,
                            X_test: torch.Tensor,
                            y_test: torch.Tensor,
                            epochs: int = 100,
                            model_name: str = "Model") -> Dict:
        """Train a standard neural network."""
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    val_pred = val_outputs.argmax(dim=1)
                    val_acc = (val_pred == y_val).float().mean().item()
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        self.logger.info(f"{model_name} early stopping at epoch {epoch}")
                        break
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train).argmax(dim=1)
            val_pred = model(X_val).argmax(dim=1)
            test_pred = model(X_test).argmax(dim=1)
            
            train_acc = (train_pred == y_train).float().mean().item()
            val_acc = (val_pred == y_val).float().mean().item()
            test_acc = (test_pred == y_test).float().mean().item()
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'best_val_accuracy': best_val_acc,
            'model': model
        }
    
    def _train_hierarchical_nn(self, model: HierarchicalNeuralNetwork, epochs: int = 100) -> Dict:
        """Train hierarchical neural network."""
        
        # Prepare hierarchical data
        X_train = torch.FloatTensor(self.data_splits['train']['pathway_data']).to(self.device)
        y_train = {
            'organ': torch.LongTensor(self.data_splits['train']['organ_labels']).to(self.device),
            'histology': torch.LongTensor(self.data_splits['train']['histology_labels']).to(self.device),
            'molecular': torch.LongTensor(self.data_splits['train']['molecular_labels']).to(self.device)
        }
        
        X_val = torch.FloatTensor(self.data_splits['val']['pathway_data']).to(self.device)
        y_val = {
            'organ': torch.LongTensor(self.data_splits['val']['organ_labels']).to(self.device),
            'histology': torch.LongTensor(self.data_splits['val']['histology_labels']).to(self.device),
            'molecular': torch.LongTensor(self.data_splits['val']['molecular_labels']).to(self.device)
        }
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_train)
            
            # Hierarchical loss
            total_loss = 0
            for level, weight in zip(['molecular', 'histology', 'organ'], [1.0, 0.7, 0.5]):
                loss = F.cross_entropy(outputs[level], y_train[level])
                total_loss += weight * loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    val_pred = val_outputs['molecular'].argmax(dim=1)
                    val_acc = (val_pred == y_val['molecular']).float().mean().item()
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train)
            val_outputs = model(X_val)
            
            train_acc = (train_outputs['molecular'].argmax(dim=1) == y_train['molecular']).float().mean().item()
            val_acc = (val_outputs['molecular'].argmax(dim=1) == y_val['molecular']).float().mean().item()
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'best_val_accuracy': best_val_acc,
            'model': model
        }
    
    def evaluate_prototypical_networks(self, n_episodes: int = 1000) -> Dict:
        """Evaluate prototypical networks for few-shot comparison."""
        self.logger.info("Evaluating prototypical networks...")
        
        from ..data.preprocessing import MetaLearningDataLoader
        
        # Create prototypical network
        proto_net = PrototypicalNetwork(
            input_dim=self.X_train.shape[1]
        ).to(self.device)
        
        # Train prototypical network
        optimizer = torch.optim.Adam(proto_net.parameters(), lr=0.001)
        
        # Create meta-learning data loader for training
        train_loader = MetaLearningDataLoader(
            pathway_data=self.data_splits['train']['pathway_data'],
            cancer_types=self.data_splits['train']['cancer_types'],
            hierarchical_labels={
                'molecular': self.data_splits['train']['molecular_labels']
            },
            n_way=5, k_shot=5, n_query=15
        )
        
        # Training episodes
        for episode in range(1000):
            proto_net.train()
            optimizer.zero_grad()
            
            task = train_loader.sample_task()
            support_x, support_y = task['support']
            query_x, query_y = task['query']
            
            support_x = support_x.to(self.device)
            support_y = support_y['molecular'].to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y['molecular'].to(self.device)
            
            logits = proto_net(support_x, support_y, query_x)
            loss = F.cross_entropy(logits, query_y)
            
            loss.backward()
            optimizer.step()
        
        # Evaluation
        proto_net.eval()
        val_loader = MetaLearningDataLoader(
            pathway_data=self.data_splits['val']['pathway_data'],
            cancer_types=self.data_splits['val']['cancer_types'],
            hierarchical_labels={
                'molecular': self.data_splits['val']['molecular_labels']
            },
            n_way=5, k_shot=5, n_query=15
        )
        
        accuracies = []
        for _ in range(100):
            task = val_loader.sample_task()
            support_x, support_y = task['support']
            query_x, query_y = task['query']
            
            support_x = support_x.to(self.device)
            support_y = support_y['molecular'].to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y['molecular'].to(self.device)
            
            with torch.no_grad():
                logits = proto_net(support_x, support_y, query_x)
                pred = logits.argmax(dim=1)
                acc = (pred == query_y).float().mean().item()
                accuracies.append(acc)
        
        proto_results = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'model': proto_net
        }
        
        self.results['prototypical_networks'] = proto_results
        return proto_results
    
    def compare_all_baselines(self) -> Dict:
        """Run comprehensive baseline comparison."""
        self.logger.info("Running comprehensive baseline comparison...")
        
        # Run all evaluations
        sklearn_results = self.evaluate_sklearn_baselines()
        nn_results = self.evaluate_neural_networks()
        proto_results = self.evaluate_prototypical_networks()
        
        # Compile summary
        summary = {
            'sklearn_best': max(sklearn_results.items(), key=lambda x: x[1]['val_accuracy']),
            'neural_network_best': max(nn_results.items(), key=lambda x: x[1]['val_accuracy']),
            'prototypical_accuracy': proto_results['mean_accuracy']
        }
        
        self.logger.info("Baseline comparison completed!")
        self.logger.info(f"Best sklearn model: {summary['sklearn_best'][0]} ({summary['sklearn_best'][1]['val_accuracy']:.4f})")
        self.logger.info(f"Best neural network: {summary['neural_network_best'][0]} ({summary['neural_network_best'][1]['val_accuracy']:.4f})")
        self.logger.info(f"Prototypical networks: {summary['prototypical_accuracy']:.4f}")
        
        return {
            'individual_results': self.results,
            'summary': summary
        }