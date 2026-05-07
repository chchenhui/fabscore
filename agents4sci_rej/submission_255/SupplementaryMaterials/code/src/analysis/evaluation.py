"""
Comprehensive Evaluation Framework for Hierarchical Meta-Learning
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

from ..models.hierarchical_maml import HierarchicalMAML, MetaLearner
from ..data.preprocessing import MetaLearningDataLoader


class HierarchicalEvaluator:
    """
    Comprehensive evaluation framework for hierarchical meta-learning models.
    """
    
    def __init__(self, 
                 model: HierarchicalMAML,
                 meta_learner: MetaLearner,
                 data_splits: Dict,
                 hierarchy_mapping: Dict,
                 device: str = 'cuda',
                 results_dir: str = './results'):
        
        self.model = model
        self.meta_learner = meta_learner
        self.data_splits = data_splits
        self.hierarchy_mapping = hierarchy_mapping
        self.device = device
        
        # Setup results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.evaluation_results = {}
        
    def comprehensive_evaluation(self) -> Dict:
        """
        Run comprehensive evaluation including few-shot scenarios,
        transferability analysis, and pathway importance.
        """
        self.logger.info("Starting comprehensive evaluation...")
        
        # Few-shot evaluation
        few_shot_results = self.evaluate_few_shot_scenarios()
        
        # Cross-cancer transferability
        transferability_results = self.evaluate_transferability()
        
        # Pathway importance analysis
        importance_results = self.analyze_pathway_importance()
        
        # Hierarchical performance analysis
        hierarchical_results = self.analyze_hierarchical_performance()
        
        # Compile all results
        self.evaluation_results = {
            'few_shot': few_shot_results,
            'transferability': transferability_results,
            'pathway_importance': importance_results,
            'hierarchical_performance': hierarchical_results
        }
        
        # Save results
        self._save_results()
        
        self.logger.info("Comprehensive evaluation completed!")
        return self.evaluation_results
    
    def evaluate_few_shot_scenarios(self) -> Dict:
        """Evaluate model performance across different few-shot scenarios."""
        self.logger.info("Evaluating few-shot scenarios...")
        
        shot_scenarios = [1, 3, 5, 10]
        query_sizes = [15, 20, 25]
        
        results = {}
        
        for k_shot in shot_scenarios:
            for n_query in query_sizes:
                scenario_name = f"{k_shot}_shot_{n_query}_query"
                self.logger.info(f"Evaluating {scenario_name}...")
                
                # Create test loader for this scenario
                test_loader = MetaLearningDataLoader(
                    pathway_data=self.data_splits['test']['pathway_data'],
                    cancer_types=self.data_splits['test']['cancer_types'],
                    hierarchical_labels={
                        'organ': self.data_splits['test']['organ_labels'],
                        'histology': self.data_splits['test']['histology_labels'],
                        'molecular': self.data_splits['test']['molecular_labels']
                    },
                    n_way=5,
                    k_shot=k_shot,
                    n_query=n_query,
                    n_tasks_per_batch=1
                )
                
                # Run evaluation episodes
                accuracies = {'organ': [], 'histology': [], 'molecular': []}
                confidences = []
                adaptation_curves = []
                
                n_episodes = 200
                for episode in range(n_episodes):
                    task = test_loader.sample_task()
                    episode_results = self._evaluate_single_task(task)
                    
                    for level in accuracies:
                        accuracies[level].append(episode_results['accuracies'][level])
                    
                    confidences.append(episode_results['confidence'])
                    adaptation_curves.append(episode_results['adaptation_curve'])
                
                # Compute statistics
                scenario_results = {}
                for level in accuracies:
                    acc_array = np.array(accuracies[level])
                    scenario_results[f'{level}_accuracy'] = {
                        'mean': np.mean(acc_array),
                        'std': np.std(acc_array),
                        'ci_95': 1.96 * np.std(acc_array) / np.sqrt(len(acc_array)),
                        'median': np.median(acc_array),
                        'min': np.min(acc_array),
                        'max': np.max(acc_array)
                    }
                
                scenario_results['confidence'] = {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences)
                }
                
                scenario_results['adaptation_curve'] = {
                    'mean_curve': np.mean(adaptation_curves, axis=0).tolist(),
                    'std_curve': np.std(adaptation_curves, axis=0).tolist()
                }
                
                results[scenario_name] = scenario_results
        
        return results
    
    def _evaluate_single_task(self, task: Dict) -> Dict:
        """Evaluate model on a single meta-learning task."""
        support_x, support_y = task['support']
        query_x, query_y = task['query']
        
        # Move to device
        support_x = support_x.to(self.device)
        query_x = query_x.to(self.device)
        for level in support_y:
            support_y[level] = support_y[level].to(self.device)
            query_y[level] = query_y[level].to(self.device)
        
        # Track adaptation curve
        adaptation_curve = []
        
        # Fast adaptation with tracking
        adapted_model = self._fast_adapt_with_tracking(
            support_x, support_y, query_x, query_y, adaptation_curve
        )
        
        # Final evaluation
        adapted_model.eval()
        with torch.no_grad():
            query_predictions, _ = adapted_model(query_x)
            
            # Compute accuracies for each level
            accuracies = {}
            for level in ['organ', 'histology', 'molecular']:
                pred_labels = query_predictions[level].argmax(dim=1)
                accuracy = (pred_labels == query_y[level]).float().mean().item()
                accuracies[level] = accuracy
            
            # Compute confidence (average max probability)
            molecular_probs = torch.softmax(query_predictions['molecular'], dim=1)
            confidence = molecular_probs.max(dim=1)[0].mean().item()
        
        return {
            'accuracies': accuracies,
            'confidence': confidence,
            'adaptation_curve': adaptation_curve
        }
    
    def _fast_adapt_with_tracking(self, 
                                support_x: torch.Tensor,
                                support_y: Dict,
                                query_x: torch.Tensor,
                                query_y: Dict,
                                adaptation_curve: List) -> nn.Module:
        """Fast adaptation with performance tracking."""
        
        # Clone model for adaptation
        adapted_model = type(self.model)(
            input_dim=self.model.encoder.input_dim,
            feature_dim=self.model.encoder.output_dim,
            num_organ_classes=self.model.num_classes['organ'],
            num_histology_classes=self.model.num_classes['histology'],
            num_molecular_classes=self.model.num_classes['molecular']
        ).to(self.device)
        
        adapted_model.load_state_dict(self.model.state_dict())
        
        # Adaptation optimizer
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.meta_learner.inner_lr)
        
        # Adaptation steps with tracking
        for step in range(self.meta_learner.inner_steps):
            adapted_model.train()
            optimizer.zero_grad()
            
            # Support set forward pass
            predictions, _ = adapted_model(support_x)
            
            # Compute loss
            from ..models.hierarchical_maml import hierarchical_loss
            loss, _ = hierarchical_loss(predictions, support_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track performance on query set
            adapted_model.eval()
            with torch.no_grad():
                query_predictions, _ = adapted_model(query_x)
                query_acc = (query_predictions['molecular'].argmax(dim=1) == query_y['molecular']).float().mean().item()
                adaptation_curve.append(query_acc)
        
        return adapted_model
    
    def evaluate_transferability(self) -> Dict:
        """Evaluate cross-cancer transferability."""
        self.logger.info("Evaluating cross-cancer transferability...")
        
        test_cancer_types = np.unique(self.data_splits['test']['cancer_types'])
        
        # Create transfer matrix
        transfer_matrix = np.zeros((len(test_cancer_types), len(test_cancer_types)))
        transfer_details = {}
        
        for i, source_cancer in enumerate(test_cancer_types):
            for j, target_cancer in enumerate(test_cancer_types):
                if i != j:  # Skip self-transfer
                    transfer_score = self._evaluate_cancer_transfer(source_cancer, target_cancer)
                    transfer_matrix[i, j] = transfer_score
                    
                    transfer_details[f"{source_cancer}_to_{target_cancer}"] = transfer_score
        
        # Analyze transfer patterns
        transfer_analysis = self._analyze_transfer_patterns(transfer_matrix, test_cancer_types)
        
        return {
            'transfer_matrix': transfer_matrix.tolist(),
            'cancer_types': test_cancer_types.tolist(),
            'transfer_details': transfer_details,
            'transfer_analysis': transfer_analysis
        }
    
    def _evaluate_cancer_transfer(self, source_cancer: str, target_cancer: str) -> float:
        """Evaluate transfer from source to target cancer type."""
        
        # Get source cancer data for training
        source_mask = self.data_splits['test']['cancer_types'] == source_cancer
        source_data = self.data_splits['test']['pathway_data'][source_mask]
        source_labels = self.data_splits['test']['molecular_labels'][source_mask]
        
        # Get target cancer data for evaluation
        target_mask = self.data_splits['test']['cancer_types'] == target_cancer
        target_data = self.data_splits['test']['pathway_data'][target_mask]
        target_labels = self.data_splits['test']['molecular_labels'][target_mask]
        
        if len(source_data) < 10 or len(target_data) < 10:
            return 0.0  # Insufficient data
        
        # Create few-shot learning scenario
        n_support = min(5, len(source_data) // 2)
        n_query = min(15, len(target_data) - 5)
        
        # Sample support from source, query from target
        source_indices = np.random.choice(len(source_data), n_support, replace=False)
        target_indices = np.random.choice(len(target_data), n_query + 5, replace=False)
        
        support_x = torch.FloatTensor(source_data[source_indices]).to(self.device)
        support_y = torch.LongTensor([0] * n_support).to(self.device)  # Single class
        
        query_x = torch.FloatTensor(target_data[target_indices[:n_query]]).to(self.device)
        query_y = torch.LongTensor([1] * n_query).to(self.device)  # Different class
        
        # Create binary classification task
        combined_x = torch.cat([support_x, query_x])
        combined_y = torch.cat([support_y, query_y])
        
        # Fast adaptation
        adapted_model = self.meta_learner.fast_adapt(
            combined_x[:n_support + 5], 
            {'molecular': combined_y[:n_support + 5]},
            num_steps=3
        )
        
        # Evaluate
        adapted_model.eval()
        with torch.no_grad():
            predictions, _ = adapted_model(combined_x[n_support + 5:])
            binary_predictions = predictions['molecular'][:, :2]  # Use first 2 classes
            pred_labels = binary_predictions.argmax(dim=1)
            accuracy = (pred_labels == combined_y[n_support + 5:]).float().mean().item()
        
        return accuracy
    
    def _analyze_transfer_patterns(self, transfer_matrix: np.ndarray, cancer_types: List[str]) -> Dict:
        """Analyze patterns in the transfer matrix."""
        
        # Compute transfer statistics
        non_zero_transfers = transfer_matrix[transfer_matrix > 0]
        
        analysis = {
            'mean_transfer_score': np.mean(non_zero_transfers),
            'std_transfer_score': np.std(non_zero_transfers),
            'best_source_cancers': [],
            'best_target_cancers': [],
            'worst_transfers': []
        }
        
        # Best source cancers (transfer well to others)
        source_means = np.mean(transfer_matrix, axis=1)
        best_sources = np.argsort(source_means)[-3:]
        analysis['best_source_cancers'] = [(cancer_types[i], source_means[i]) for i in best_sources]
        
        # Best target cancers (receive transfers well)
        target_means = np.mean(transfer_matrix, axis=0)
        best_targets = np.argsort(target_means)[-3:]
        analysis['best_target_cancers'] = [(cancer_types[i], target_means[i]) for i in best_targets]
        
        # Worst transfer pairs
        flat_indices = np.argsort(transfer_matrix.flatten())[:5]
        for flat_idx in flat_indices:
            i, j = np.unravel_index(flat_idx, transfer_matrix.shape)
            if i != j:  # Skip diagonal
                analysis['worst_transfers'].append(
                    (cancer_types[i], cancer_types[j], transfer_matrix[i, j])
                )
        
        return analysis
    
    def analyze_pathway_importance(self) -> Dict:
        """Analyze pathway importance using multiple methods."""
        self.logger.info("Analyzing pathway importance...")
        
        from ..models.pathway_encoder import PathwayImportanceAnalyzer
        
        # Initialize analyzer
        pathway_names = [f"Pathway_{i}" for i in range(self.data_splits['test']['pathway_data'].shape[1])]
        analyzer = PathwayImportanceAnalyzer(self.model, pathway_names)
        
        # Sample test data
        test_data = torch.FloatTensor(self.data_splits['test']['pathway_data'][:100]).to(self.device)
        test_labels = torch.LongTensor(self.data_splits['test']['molecular_labels'][:100]).to(self.device)
        
        # Compute importance using different methods
        importance_results = {}
        
        # Integrated gradients
        self.logger.info("Computing integrated gradients...")
        ig_importance = analyzer.integrated_gradients(test_data, test_labels)
        importance_results['integrated_gradients'] = {
            'scores': ig_importance.mean(dim=0).cpu().numpy().tolist(),
            'std': ig_importance.std(dim=0).cpu().numpy().tolist()
        }
        
        # Permutation importance
        self.logger.info("Computing permutation importance...")
        perm_importance = analyzer.permutation_importance(test_data, test_labels)
        importance_results['permutation'] = {
            'scores': perm_importance.cpu().numpy().tolist()
        }
        
        # Attention weights (if available)
        if hasattr(self.model.encoder, 'pathway_attention'):
            self.logger.info("Extracting attention weights...")
            attention_weights = self._extract_attention_weights(test_data)
            importance_results['attention_weights'] = {
                'scores': attention_weights.tolist()
            }
        
        # Rank correlation between methods
        correlations = self._compute_importance_correlations(importance_results)
        importance_results['correlations'] = correlations
        
        return importance_results
    
    def _extract_attention_weights(self, data: torch.Tensor) -> np.ndarray:
        """Extract attention weights from the model."""
        self.model.eval()
        
        with torch.no_grad():
            _, attention_weights = self.model.encoder(data)
            if attention_weights is not None:
                # Average attention weights across samples and heads
                avg_attention = attention_weights.mean(dim=0).mean(dim=0)
                return avg_attention.cpu().numpy()
        
        return np.zeros(data.size(1))
    
    def _compute_importance_correlations(self, importance_results: Dict) -> Dict:
        """Compute correlations between different importance methods."""
        
        methods = list(importance_results.keys())
        correlations = {}
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j:
                    scores1 = np.array(importance_results[method1]['scores'])
                    scores2 = np.array(importance_results[method2]['scores'])
                    
                    # Spearman correlation (rank-based)
                    corr, p_value = stats.spearmanr(scores1, scores2)
                    
                    correlations[f"{method1}_vs_{method2}"] = {
                        'correlation': corr,
                        'p_value': p_value
                    }
        
        return correlations
    
    def analyze_hierarchical_performance(self) -> Dict:
        """Analyze performance across hierarchy levels."""
        self.logger.info("Analyzing hierarchical performance...")
        
        # Create test loader
        test_loader = MetaLearningDataLoader(
            pathway_data=self.data_splits['test']['pathway_data'],
            cancer_types=self.data_splits['test']['cancer_types'],
            hierarchical_labels={
                'organ': self.data_splits['test']['organ_labels'],
                'histology': self.data_splits['test']['histology_labels'],
                'molecular': self.data_splits['test']['molecular_labels']
            },
            n_way=5, k_shot=5, n_query=15, n_tasks_per_batch=1
        )
        
        # Collect predictions for each level
        level_predictions = {'organ': [], 'histology': [], 'molecular': []}
        level_targets = {'organ': [], 'histology': [], 'molecular': []}
        
        n_tasks = 100
        for _ in range(n_tasks):
            task = test_loader.sample_task()
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
            
            # Predictions
            adapted_model.eval()
            with torch.no_grad():
                predictions, _ = adapted_model(query_x)
                
                for level in ['organ', 'histology', 'molecular']:
                    pred_labels = predictions[level].argmax(dim=1)
                    level_predictions[level].extend(pred_labels.cpu().numpy())
                    level_targets[level].extend(query_y[level].cpu().numpy())
        
        # Compute hierarchical consistency
        consistency_analysis = self._analyze_hierarchical_consistency(
            level_predictions, level_targets
        )
        
        # Performance per level
        level_performance = {}
        for level in ['organ', 'histology', 'molecular']:
            accuracy = np.mean(np.array(level_predictions[level]) == np.array(level_targets[level]))
            level_performance[level] = {
                'accuracy': accuracy,
                'n_predictions': len(level_predictions[level])
            }
        
        return {
            'level_performance': level_performance,
            'hierarchical_consistency': consistency_analysis
        }
    
    def _analyze_hierarchical_consistency(self, predictions: Dict, targets: Dict) -> Dict:
        """Analyze consistency across hierarchy levels."""
        
        # Convert to numpy arrays
        organ_pred = np.array(predictions['organ'])
        histology_pred = np.array(predictions['histology'])
        molecular_pred = np.array(predictions['molecular'])
        
        organ_true = np.array(targets['organ'])
        histology_true = np.array(targets['histology'])
        molecular_true = np.array(targets['molecular'])
        
        # Consistency metrics
        total_samples = len(organ_pred)
        
        # Perfect hierarchical prediction (all levels correct)
        perfect_hierarchy = ((organ_pred == organ_true) & 
                           (histology_pred == histology_true) & 
                           (molecular_pred == molecular_true))
        perfect_rate = np.mean(perfect_hierarchy)
        
        # Partial consistency rates
        organ_histology_consistent = (organ_pred == organ_true) & (histology_pred == histology_true)
        organ_molecular_consistent = (organ_pred == organ_true) & (molecular_pred == molecular_true)
        histology_molecular_consistent = (histology_pred == histology_true) & (molecular_pred == molecular_true)
        
        return {
            'perfect_hierarchy_rate': perfect_rate,
            'organ_histology_consistency': np.mean(organ_histology_consistent),
            'organ_molecular_consistency': np.mean(organ_molecular_consistent),
            'histology_molecular_consistency': np.mean(histology_molecular_consistent),
            'total_samples': total_samples
        }
    
    def _save_results(self):
        """Save all evaluation results to files."""
        
        # Save JSON results
        results_file = self.results_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
        
        # Save summary report
        self._generate_summary_report()
    
    def _generate_summary_report(self):
        """Generate a human-readable summary report."""
        
        report_file = self.results_dir / 'evaluation_summary.txt'
        
        with open(report_file, 'w') as f:
            f.write("HIERARCHICAL META-LEARNING EVALUATION SUMMARY\\n")
            f.write("=" * 50 + "\\n\\n")
            
            # Few-shot performance
            f.write("FEW-SHOT PERFORMANCE:\\n")
            f.write("-" * 20 + "\\n")
            
            if 'few_shot' in self.evaluation_results:
                for scenario, results in self.evaluation_results['few_shot'].items():
                    f.write(f"{scenario}:\\n")
                    f.write(f"  Molecular Accuracy: {results['molecular_accuracy']['mean']:.4f} ± {results['molecular_accuracy']['std']:.4f}\\n")
                    f.write(f"  Confidence: {results['confidence']['mean']:.4f}\\n\\n")
            
            # Pathway importance
            f.write("PATHWAY IMPORTANCE (TOP 10):\\n")
            f.write("-" * 30 + "\\n")
            
            if 'pathway_importance' in self.evaluation_results:
                ig_scores = self.evaluation_results['pathway_importance']['integrated_gradients']['scores']
                top_pathways = np.argsort(ig_scores)[-10:][::-1]
                
                for i, pathway_idx in enumerate(top_pathways):
                    f.write(f"{i+1:2d}. Pathway_{pathway_idx:2d}: {ig_scores[pathway_idx]:.4f}\\n")
            
            f.write("\\n")
            
            # Transferability summary
            f.write("TRANSFERABILITY ANALYSIS:\\n")
            f.write("-" * 25 + "\\n")
            
            if 'transferability' in self.evaluation_results:
                transfer_analysis = self.evaluation_results['transferability']['transfer_analysis']
                f.write(f"Mean Transfer Score: {transfer_analysis['mean_transfer_score']:.4f}\\n")
                
                f.write("\\nBest Source Cancers:\\n")
                for cancer, score in transfer_analysis['best_source_cancers']:
                    f.write(f"  {cancer}: {score:.4f}\\n")
        
        self.logger.info(f"Summary report saved to {report_file}")