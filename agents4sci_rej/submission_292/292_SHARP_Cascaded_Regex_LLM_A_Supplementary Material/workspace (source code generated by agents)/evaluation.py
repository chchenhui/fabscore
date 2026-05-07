"""
Evaluation module for phishing detection methods
Calculates various performance metrics
"""

import numpy as np
import logging

# Try to import sklearn metrics
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, confusion_matrix, roc_auc_score,
        classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available. Using custom metric implementations.")

logger = logging.getLogger(__name__)

# Custom metric implementations if sklearn not available
if not SKLEARN_AVAILABLE:
    def accuracy_score(y_true, y_pred):
        """Custom accuracy score implementation"""
        return np.mean(y_true == y_pred)
    
    def precision_score(y_true, y_pred, zero_division=0):
        """Custom precision score implementation"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        if tp + fp == 0:
            return zero_division
        return tp / (tp + fp)
    
    def recall_score(y_true, y_pred, zero_division=0):
        """Custom recall score implementation"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        if tp + fn == 0:
            return zero_division
        return tp / (tp + fn)
    
    def f1_score(y_true, y_pred, zero_division=0):
        """Custom F1 score implementation"""
        prec = precision_score(y_true, y_pred, zero_division)
        rec = recall_score(y_true, y_pred, zero_division)
        if prec + rec == 0:
            return zero_division
        return 2 * (prec * rec) / (prec + rec)
    
    def confusion_matrix(y_true, y_pred):
        """Custom confusion matrix implementation"""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return np.array([[tn, fp], [fn, tp]])

class Evaluator:
    """Evaluate phishing detection methods"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_method(self, method, test_data, method_name):
        """Evaluate a single detection method"""
        
        # Get predictions
        predictions = method.predict(test_data)
        
        # Get true labels
        true_labels = np.array([email['label'] for email in test_data])
        
        # Calculate metrics
        metrics = self.calculate_metrics(true_labels, predictions)
        
        # Store results
        self.results[method_name] = metrics
        
        return metrics
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # True/False Positives/Negatives
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Additional metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics
    
    def compare_methods(self):
        """Compare all evaluated methods"""
        
        if not self.results:
            logger.warning("No results to compare")
            return None
        
        comparison = []
        
        for method_name, metrics in self.results.items():
            comparison.append({
                'method': method_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            })
        
        # Sort by F1 score
        comparison.sort(key=lambda x: x['f1_score'], reverse=True)
        
        return comparison
    
    def print_detailed_report(self, method_name):
        """Print detailed report for a method"""
        
        if method_name not in self.results:
            logger.error(f"No results for {method_name}")
            return
        
        metrics = self.results[method_name]
        
        print(f"\n{'='*60}")
        print(f"Detailed Report: {method_name}")
        print(f"{'='*60}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:    {metrics['accuracy']:.3f}")
        print(f"  Precision:   {metrics['precision']:.3f}")
        print(f"  Recall:      {metrics['recall']:.3f}")
        print(f"  F1-Score:    {metrics['f1_score']:.3f}")
        
        if 'specificity' in metrics:
            print(f"  Specificity: {metrics['specificity']:.3f}")
            print(f"  FPR:         {metrics['false_positive_rate']:.3f}")
            print(f"  FNR:         {metrics['false_negative_rate']:.3f}")
        
        if 'confusion_matrix' in metrics:
            print(f"\nConfusion Matrix:")
            cm = metrics['confusion_matrix']
            if len(cm) == 2:
                print(f"                 Predicted")
                print(f"                 Neg    Pos")
                print(f"  Actual Neg    {cm[0][0]:4d}   {cm[0][1]:4d}")
                print(f"  Actual Pos    {cm[1][0]:4d}   {cm[1][1]:4d}")
        
        print(f"\n{'='*60}")

class CrossValidator:
    """Perform cross-validation for robust evaluation"""
    
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
    
    def cross_validate(self, method, data, method_name):
        """Perform k-fold cross-validation"""
        
        # Shuffle data
        import random
        random.shuffle(data)
        
        # Split into folds
        fold_size = len(data) // self.n_folds
        folds = []
        
        for i in range(self.n_folds):
            start = i * fold_size
            end = start + fold_size if i < self.n_folds - 1 else len(data)
            folds.append(data[start:end])
        
        # Evaluate on each fold
        fold_results = []
        
        for i in range(self.n_folds):
            # Prepare train and test sets
            test_fold = folds[i]
            train_folds = [fold for j, fold in enumerate(folds) if j != i]
            train_data = [item for fold in train_folds for item in fold]
            
            # Train if method has fit method
            if hasattr(method, 'fit'):
                method.fit(train_data)
            
            # Evaluate
            predictions = method.predict(test_fold)
            true_labels = np.array([email['label'] for email in test_fold])
            
            # Calculate metrics
            fold_metrics = {
                'accuracy': accuracy_score(true_labels, predictions),
                'precision': precision_score(true_labels, predictions, zero_division=0),
                'recall': recall_score(true_labels, predictions, zero_division=0),
                'f1_score': f1_score(true_labels, predictions, zero_division=0)
            }
            
            fold_results.append(fold_metrics)
        
        # Average results
        avg_results = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            values = [fold[metric] for fold in fold_results]
            avg_results[f'avg_{metric}'] = np.mean(values)
            avg_results[f'std_{metric}'] = np.std(values)
        
        logger.info(f"Cross-validation results for {method_name}:")
        logger.info(f"  Avg Accuracy:  {avg_results['avg_accuracy']:.3f} ± {avg_results['std_accuracy']:.3f}")
        logger.info(f"  Avg Precision: {avg_results['avg_precision']:.3f} ± {avg_results['std_precision']:.3f}")
        logger.info(f"  Avg Recall:    {avg_results['avg_recall']:.3f} ± {avg_results['std_recall']:.3f}")
        logger.info(f"  Avg F1-Score:  {avg_results['avg_f1_score']:.3f} ± {avg_results['std_f1_score']:.3f}")
        
        return avg_results