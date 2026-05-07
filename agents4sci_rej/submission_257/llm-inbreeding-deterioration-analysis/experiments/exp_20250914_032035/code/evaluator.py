"""
Evaluator for LLM Inbreeding Deterioration Analysis

This module implements comprehensive evaluation metrics to assess 
model performance degradation across generations and conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from transformers import AutoTokenizer
from collections import Counter
import re
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InbreedingEvaluator:
    """Comprehensive evaluator for multi-generation model performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"])
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.results = []
        self.results_dir = Path(config["paths"]["results_dir"])
        self.results_dir.mkdir(exist_ok=True)
    
    def evaluate_generation(self, 
                          predictions: List[str], 
                          references: List[str],
                          generation: int,
                          condition: str,
                          task_types: List[str]) -> Dict[str, float]:
        """Evaluate a specific generation and condition."""
        
        logger.info(f"Evaluating Generation {generation}, Condition: {condition}")
        
        # Calculate all metrics
        metrics = {}
        
        # Language Quality Metrics
        metrics.update(self._calculate_language_quality(predictions))
        
        # Factual Accuracy Metrics
        metrics.update(self._calculate_factual_accuracy(predictions, references))
        
        # Diversity Metrics
        metrics.update(self._calculate_diversity_metrics(predictions))
        
        # Coherence Metrics
        metrics.update(self._calculate_coherence_metrics(predictions, references))
        
        # Reasoning Quality Metrics
        metrics.update(self._calculate_reasoning_metrics(predictions, references, task_types))
        
        # Creativity Metrics
        metrics.update(self._calculate_creativity_metrics(predictions, task_types))
        
        # Add metadata
        metrics['generation'] = generation
        metrics['condition'] = condition
        metrics['num_samples'] = len(predictions)
        
        # Store results
        self.results.append(metrics)
        
        logger.info(f"Evaluation complete. Key metrics: "
                   f"Perplexity: {metrics.get('perplexity', 'N/A'):.3f}, "
                   f"Diversity: {metrics.get('distinct_2_grams', 'N/A'):.3f}, "
                   f"Coherence: {metrics.get('coherence_score', 'N/A'):.3f}")
        
        return metrics
    
    def _calculate_language_quality(self, predictions: List[str]) -> Dict[str, float]:
        """Calculate language quality metrics."""
        metrics = {}
        
        # Simulated perplexity (in real implementation, would use actual language model)
        lengths = [len(pred.split()) for pred in predictions]
        avg_length = np.mean(lengths)
        
        # Simple proxy for perplexity based on repetition and length
        repetition_penalty = self._calculate_repetition_penalty(predictions)
        metrics['perplexity'] = max(10.0, 50.0 + repetition_penalty - avg_length * 0.1)
        
        # Fluency score (simplified - would use actual fluency model in practice)
        metrics['fluency_score'] = max(0.1, 1.0 - repetition_penalty / 100.0)
        
        # Average sentence length
        metrics['avg_sentence_length'] = avg_length
        
        return metrics
    
    def _calculate_repetition_penalty(self, predictions: List[str]) -> float:
        """Calculate repetition penalty score."""
        total_penalty = 0
        
        for pred in predictions:
            words = pred.lower().split()
            if len(words) > 0:
                word_counts = Counter(words)
                # Penalty for repeated words
                repetition_ratio = sum(count - 1 for count in word_counts.values()) / len(words)
                total_penalty += repetition_ratio * 100
        
        return total_penalty / max(1, len(predictions))
    
    def _calculate_factual_accuracy(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate factual accuracy metrics."""
        metrics = {}
        
        exact_matches = 0
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            # Exact match (case-insensitive)
            if pred.lower().strip() == ref.lower().strip():
                exact_matches += 1
            
            # Token-level F1 score
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if len(ref_tokens) == 0:
                f1 = 1.0 if len(pred_tokens) == 0 else 0.0
            else:
                precision = len(pred_tokens & ref_tokens) / max(1, len(pred_tokens))
                recall = len(pred_tokens & ref_tokens) / len(ref_tokens)
                f1 = 2 * precision * recall / max(1, precision + recall)
            
            f1_scores.append(f1)
        
        metrics['exact_match'] = exact_matches / max(1, len(predictions))
        metrics['f1_score'] = np.mean(f1_scores)
        
        return metrics
    
    def _calculate_diversity_metrics(self, predictions: List[str]) -> Dict[str, float]:
        """Calculate text diversity metrics."""
        metrics = {}
        
        # Combine all predictions
        all_text = " ".join(predictions).lower()
        tokens = all_text.split()
        
        if len(tokens) == 0:
            metrics['distinct_1_grams'] = 0.0
            metrics['distinct_2_grams'] = 0.0
            metrics['entropy'] = 0.0
            return metrics
        
        # Distinct n-grams
        unigrams = tokens
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
        
        metrics['distinct_1_grams'] = len(set(unigrams)) / max(1, len(unigrams))
        metrics['distinct_2_grams'] = len(set(bigrams)) / max(1, len(bigrams))
        
        # Entropy calculation
        token_counts = Counter(tokens)
        total_tokens = sum(token_counts.values())
        probabilities = [count / total_tokens for count in token_counts.values()]
        metrics['entropy'] = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        return metrics
    
    def _calculate_coherence_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate coherence and semantic similarity metrics."""
        metrics = {}
        
        # Simplified coherence score based on sentence structure
        coherence_scores = []
        
        for pred in predictions:
            sentences = pred.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= 1:
                coherence_scores.append(0.5)  # Neutral score for single sentences
            else:
                # Simple coherence heuristic: consistent sentence length and structure
                lengths = [len(s.split()) for s in sentences]
                if len(lengths) > 0:
                    coherence = 1.0 - (np.std(lengths) / max(1, np.mean(lengths)))
                    coherence_scores.append(max(0.0, min(1.0, coherence)))
                else:
                    coherence_scores.append(0.0)
        
        metrics['coherence_score'] = np.mean(coherence_scores)
        
        # Semantic similarity (simplified - would use embeddings in practice)
        similarity_scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            if len(ref_words) == 0 and len(pred_words) == 0:
                similarity = 1.0
            elif len(ref_words) == 0 or len(pred_words) == 0:
                similarity = 0.0
            else:
                jaccard_sim = len(pred_words & ref_words) / len(pred_words | ref_words)
                similarity_scores.append(jaccard_sim)
        
        metrics['semantic_similarity'] = np.mean(similarity_scores) if similarity_scores else 0.0
        
        return metrics
    
    def _calculate_reasoning_metrics(self, predictions: List[str], references: List[str], task_types: List[str]) -> Dict[str, float]:
        """Calculate reasoning quality metrics."""
        metrics = {}
        
        # Logical consistency (simplified heuristic)
        consistency_scores = []
        
        for pred, task_type in zip(predictions, task_types):
            if task_type == "question_answering":
                # Check for logical structure in answers
                score = self._assess_qa_logic(pred)
                consistency_scores.append(score)
            elif task_type == "summarization":
                # Check for summarization quality
                score = self._assess_summary_logic(pred)
                consistency_scores.append(score)
            else:
                consistency_scores.append(0.5)  # Neutral for other tasks
        
        metrics['logical_consistency'] = np.mean(consistency_scores) if consistency_scores else 0.5
        
        # Problem-solving accuracy (based on task completion)
        accuracy_scores = []
        for pred, ref, task_type in zip(predictions, references, task_types):
            if task_type == "question_answering":
                # Simple accuracy check for QA
                score = 1.0 if any(word in pred.lower() for word in ref.lower().split()[:3]) else 0.0
                accuracy_scores.append(score)
        
        metrics['problem_solving_accuracy'] = np.mean(accuracy_scores) if accuracy_scores else 0.0
        
        return metrics
    
    def _assess_qa_logic(self, answer: str) -> float:
        """Assess logical quality of question-answering response."""
        # Simple heuristics for logical structure
        score = 0.5  # Base score
        
        # Check for explanation structure
        if any(word in answer.lower() for word in ['because', 'due to', 'therefore', 'as a result']):
            score += 0.2
        
        # Check for factual claim structure
        if any(word in answer.lower() for word in ['is', 'are', 'was', 'were']):
            score += 0.1
        
        # Penalize repetition
        words = answer.lower().split()
        if len(words) > 0:
            repetition_ratio = len(words) - len(set(words))
            score -= min(0.3, repetition_ratio / len(words))
        
        return max(0.0, min(1.0, score))
    
    def _assess_summary_logic(self, summary: str) -> float:
        """Assess logical quality of summarization response."""
        score = 0.5  # Base score
        
        # Check for conciseness
        word_count = len(summary.split())
        if 10 <= word_count <= 50:  # Reasonable summary length
            score += 0.2
        
        # Check for topic coherence
        sentences = summary.split('.')
        if len(sentences) >= 1:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_creativity_metrics(self, predictions: List[str], task_types: List[str]) -> Dict[str, float]:
        """Calculate creativity and novelty metrics."""
        metrics = {}
        
        novelty_scores = []
        diversity_scores = []
        
        for pred, task_type in zip(predictions, task_types):
            if task_type == "creative_writing":
                # Assess novelty in creative tasks
                novelty = self._assess_novelty(pred)
                novelty_scores.append(novelty)
                
                # Assess semantic diversity
                diversity = self._assess_semantic_diversity(pred)
                diversity_scores.append(diversity)
            else:
                # Non-creative tasks get neutral scores
                novelty_scores.append(0.5)
                diversity_scores.append(0.5)
        
        metrics['novelty_score'] = np.mean(novelty_scores)
        metrics['semantic_diversity'] = np.mean(diversity_scores)
        
        return metrics
    
    def _assess_novelty(self, text: str) -> float:
        """Assess novelty of creative text."""
        # Simple novelty heuristics
        score = 0.5  # Base score
        
        # Check for descriptive language
        descriptive_words = ['beautiful', 'stunning', 'magnificent', 'colorful', 'bright', 'dark', 'mysterious']
        if any(word in text.lower() for word in descriptive_words):
            score += 0.2
        
        # Check for specific details
        if len(text.split()) > 20:  # Detailed response
            score += 0.1
        
        # Penalize generic phrases
        generic_phrases = ['it was', 'there was', 'very good', 'really nice']
        for phrase in generic_phrases:
            if phrase in text.lower():
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _assess_semantic_diversity(self, text: str) -> float:
        """Assess semantic diversity within text."""
        words = text.lower().split()
        if len(words) <= 1:
            return 0.0
        
        # Simple diversity based on unique words
        unique_words = len(set(words))
        diversity_ratio = unique_words / len(words)
        
        return min(1.0, diversity_ratio * 1.5)  # Scale up slightly
    
    def analyze_deterioration_patterns(self) -> Dict[str, Any]:
        """Analyze deterioration patterns across generations and conditions."""
        
        if not self.results:
            logger.warning("No results to analyze. Run evaluations first.")
            return {}
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        analysis = {
            'statistical_tests': {},
            'deterioration_rates': {},
            'condition_comparisons': {},
            'metric_correlations': {}
        }
        
        # Statistical significance tests
        conditions = df['condition'].unique()
        generations = sorted(df['generation'].unique())
        
        for condition in conditions:
            condition_data = df[df['condition'] == condition]
            
            # Test for significant degradation across generations
            metrics_to_test = ['f1_score', 'distinct_2_grams', 'coherence_score', 'logical_consistency']
            
            for metric in metrics_to_test:
                if metric in condition_data.columns:
                    # Group by generation and test for trend
                    generation_means = condition_data.groupby('generation')[metric].mean()
                    
                    if len(generation_means) >= 3:
                        # Correlation test for trend
                        corr, p_value = stats.pearsonr(generation_means.index, generation_means.values)
                        
                        analysis['statistical_tests'][f'{condition}_{metric}'] = {
                            'correlation': corr,
                            'p_value': p_value,
                            'significant_decline': p_value < 0.05 and corr < -0.5
                        }
        
        # Calculate deterioration rates
        for condition in conditions:
            condition_data = df[df['condition'] == condition]
            
            if len(condition_data) >= 2:
                first_gen = condition_data[condition_data['generation'] == condition_data['generation'].min()]
                last_gen = condition_data[condition_data['generation'] == condition_data['generation'].max()]
                
                deterioration_rates = {}
                for metric in ['f1_score', 'distinct_2_grams', 'coherence_score']:
                    if metric in first_gen.columns and metric in last_gen.columns:
                        first_val = first_gen[metric].mean()
                        last_val = last_gen[metric].mean()
                        
                        if first_val > 0:
                            deterioration_rate = (first_val - last_val) / first_val * 100
                            deterioration_rates[metric] = deterioration_rate
                
                analysis['deterioration_rates'][condition] = deterioration_rates
        
        # Condition comparisons
        if len(conditions) > 1:
            for gen in generations:
                gen_data = df[df['generation'] == gen]
                
                for metric in ['f1_score', 'distinct_2_grams', 'coherence_score']:
                    if metric in gen_data.columns:
                        condition_means = gen_data.groupby('condition')[metric].mean()
                        
                        if len(condition_means) >= 2:
                            # Statistical test between conditions
                            condition_values = [gen_data[gen_data['condition'] == cond][metric].values 
                                              for cond in condition_means.index]
                            
                            if all(len(vals) > 0 for vals in condition_values):
                                f_stat, p_value = stats.f_oneway(*condition_values)
                                
                                analysis['condition_comparisons'][f'gen_{gen}_{metric}'] = {
                                    'f_statistic': f_stat,
                                    'p_value': p_value,
                                    'significant_difference': p_value < 0.05,
                                    'condition_means': condition_means.to_dict()
                                }
        
        logger.info("Deterioration pattern analysis complete")
        return analysis
    
    def save_results(self, filename: str = "evaluation_results.json"):
        """Save all evaluation results to file."""
        results_path = self.results_dir / filename
        
        # Prepare data for JSON serialization
        serializable_results = []
        for result in self.results:
            serialized = {}
            for key, value in result.items():
                if isinstance(value, np.floating):
                    serialized[key] = float(value)
                elif isinstance(value, np.integer):
                    serialized[key] = int(value)
                else:
                    serialized[key] = value
            serializable_results.append(serialized)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def generate_visualization_report(self):
        """Generate comprehensive visualization report."""
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LLM Inbreeding Deterioration Analysis Results', fontsize=16)
        
        # Plot 1: Performance by Generation
        metrics_to_plot = ['f1_score', 'distinct_2_grams', 'coherence_score', 'logical_consistency']
        
        for i, metric in enumerate(metrics_to_plot[:4]):
            ax = axes[i//2, i%2]
            
            if metric in df.columns:
                for condition in df['condition'].unique():
                    condition_data = df[df['condition'] == condition]
                    generation_means = condition_data.groupby('generation')[metric].mean()
                    
                    ax.plot(generation_means.index, generation_means.values, 
                           marker='o', label=condition, linewidth=2)
                
                ax.set_xlabel('Generation')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.results_dir / "deterioration_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {viz_path}")

if __name__ == "__main__":
    from config import CONFIG
    
    # Example usage
    evaluator = InbreedingEvaluator(CONFIG)
    
    # Example data
    sample_predictions = [
        "The capital of France is Paris, a beautiful historic city.",
        "Photosynthesis is how plants make food from sunlight.",
        "Gravity pulls objects toward each other with force."
    ]
    
    sample_references = [
        "The capital of France is Paris, a historic city known for its culture.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "Gravity is a fundamental force that attracts objects with mass."
    ]
    
    sample_task_types = ["question_answering", "question_answering", "question_answering"]
    
    # Run evaluation
    metrics = evaluator.evaluate_generation(
        predictions=sample_predictions,
        references=sample_references,
        generation=1,
        condition="human_control",
        task_types=sample_task_types
    )
    
    print("Evaluation metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")