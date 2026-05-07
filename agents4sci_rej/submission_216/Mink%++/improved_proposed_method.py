import os, argparse, json, pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import zlib
import re
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


# helper function
def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data

# Multi-Layer Token Probability Concentration Analysis functions
def get_model_layers(model):
    """Get the total number of layers and layer access for different model architectures"""
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # Pythia-style models
        return len(model.transformer.h), model.transformer.h
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
        # Mamba-style models
        return len(model.backbone.layers), model.backbone.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Some other transformer variants
        return len(model.model.layers), model.model.layers
    else:
        # Fallback - return 0 to disable multi-layer analysis
        return 0, None

def extract_intermediate_layers(model, input_ids, layer_indices):
    """Extract hidden states from specified intermediate layers"""
    try:
        num_layers, layer_modules = get_model_layers(model)
        if num_layers == 0 or layer_modules is None:
            return None
        
        # Ensure layer indices are valid
        layer_indices = [idx for idx in layer_indices if 0 <= idx < num_layers]
        if not layer_indices:
            return None
        
        layer_outputs = {}
        
        # Forward pass with hooks to capture intermediate outputs
        hooks = {}
        
        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                hooks[layer_idx] = output if isinstance(output, torch.Tensor) else output[0]
            return hook_fn
        
        # Register hooks for selected layers
        hook_handles = []
        for layer_idx in layer_indices:
            handle = layer_modules[layer_idx].register_forward_hook(create_hook(layer_idx))
            hook_handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_ids)
        
        # Clean up hooks
        for handle in hook_handles:
            handle.remove()
        
        # Convert hooks to layer_outputs
        for layer_idx in layer_indices:
            if layer_idx in hooks:
                hidden_states = hooks[layer_idx]
                # Project to vocabulary space
                if hasattr(model, 'lm_head'):
                    logits = model.lm_head(hidden_states)
                elif hasattr(model, 'head'):
                    logits = model.head(hidden_states)
                else:
                    # Fallback - skip this layer
                    continue
                
                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)
                layer_outputs[layer_idx] = probs
        
        return layer_outputs if layer_outputs else None
        
    except Exception as e:
        print(f"Warning: Multi-layer extraction failed: {e}")
        return None

def compute_shannon_entropy(probs):
    """Calculate Shannon entropy: H(X) = -Σ p(x) log p(x)"""
    # Add small epsilon to avoid log(0)
    probs_safe = torch.clamp(probs, min=1e-12)
    log_probs = torch.log(probs_safe)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def compute_gini_coefficient(probs):
    """Calculate Gini coefficient to measure probability concentration"""
    # Sort probabilities in ascending order
    sorted_probs, _ = torch.sort(probs, dim=-1)
    n = sorted_probs.shape[-1]
    
    # Calculate Gini coefficient
    indices = torch.arange(1, n + 1, dtype=torch.float32, device=sorted_probs.device)
    indices = indices.unsqueeze(0).expand_as(sorted_probs)
    
    gini = (2.0 * indices - n - 1.0) * sorted_probs
    gini = torch.sum(gini, dim=-1)
    gini = 1.0 - gini / (n * torch.sum(sorted_probs, dim=-1))
    
    return gini

def compute_topk_concentration(probs, k_values=[5, 10, 20, 50]):
    """Calculate fraction of probability mass in top-k tokens"""
    concentrations = {}
    vocab_size = probs.shape[-1]
    
    for k in k_values:
        if k >= vocab_size:
            k = vocab_size - 1
        
        topk_probs, _ = torch.topk(probs, k, dim=-1)
        concentration = torch.sum(topk_probs, dim=-1)
        concentrations[f'top{k}'] = concentration
    
    return concentrations

def compute_effective_vocab_size(probs, threshold=0.9):
    """Number of tokens needed to capture threshold% of probability mass"""
    # Sort probabilities in descending order
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    
    # Calculate cumulative sum
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    
    # Find where cumsum exceeds threshold
    mask = cumsum >= threshold
    
    # Get the first index where threshold is exceeded for each sequence position
    first_exceed = torch.argmax(mask.float(), dim=-1) + 1
    
    # Handle cases where threshold is never reached (shouldn't happen with threshold < 1)
    max_indices = torch.full_like(first_exceed, sorted_probs.shape[-1])
    effective_size = torch.where(torch.any(mask, dim=-1), first_exceed, max_indices)
    
    # Normalize by vocabulary size
    vocab_size = probs.shape[-1]
    normalized_size = effective_size.float() / vocab_size
    
    return normalized_size

def compute_layer_concentration_features(layer_probs, layer_idx):
    """Compute all concentration metrics for a given layer"""
    features = {}
    
    # Remove batch dimension and work with sequence
    if len(layer_probs.shape) == 3:
        layer_probs = layer_probs[0]  # Remove batch dimension
    
    # Shannon entropy
    entropy = compute_shannon_entropy(layer_probs)
    features['entropy'] = entropy.mean().item()
    
    # Gini coefficient
    gini = compute_gini_coefficient(layer_probs)
    features['gini'] = gini.mean().item()
    
    # Top-k concentrations
    topk_conc = compute_topk_concentration(layer_probs)
    for k_name, k_values in topk_conc.items():
        features[k_name] = k_values.mean().item()
    
    # Effective vocabulary size
    eff_vocab = compute_effective_vocab_size(layer_probs)
    features['eff_vocab'] = eff_vocab.mean().item()
    
    return features

def compute_weighted_harmonic_mean(values, weights):
    """Compute weighted harmonic mean that's robust to outliers"""
    if not values or len(values) == 0:
        return 0.0
    
    values = np.array(values)
    weights = np.array(weights)
    
    # Handle edge cases and ensure positive values for harmonic mean
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    values = np.abs(values) + epsilon  # Ensure positive values
    
    # Compute weighted harmonic mean: HM = (Σw) / (Σ(w/x))
    try:
        weighted_reciprocals = weights / values
        harmonic_mean = np.sum(weights) / np.sum(weighted_reciprocals)
        return harmonic_mean
    except (ZeroDivisionError, ValueError):
        # Fallback to weighted arithmetic mean
        return np.average(values, weights=weights) if np.sum(weights) > 0 else np.mean(values)

def compute_sequence_coherence_penalty(probs, input_ids):
    """Compute sequence coherence penalty based on adjacent token probability consistency"""
    if len(probs.shape) == 3:
        probs = probs[0]  # Remove batch dimension
    
    if probs.shape[0] < 2:  # Need at least 2 tokens for coherence analysis
        return 0.0
    
    coherence_penalty = 0.0
    
    # 1. Probability smoothness across sequence
    # Compute variance of probability differences between adjacent positions
    prob_diffs = []
    for i in range(probs.shape[0] - 1):
        # L2 distance between adjacent probability distributions
        prob_diff = torch.norm(probs[i+1] - probs[i], p=2).item()
        prob_diffs.append(prob_diff)
    
    if prob_diffs:
        # Higher variance indicates more erratic probability patterns
        prob_variance = np.var(prob_diffs)
        smoothness_penalty = prob_variance
    else:
        smoothness_penalty = 0.0
    
    # 2. Token prediction consistency
    # Check if high-probability tokens are consistent with context
    token_consistency_penalty = 0.0
    if input_ids is not None and len(input_ids.shape) >= 1:
        actual_tokens = input_ids.squeeze() if len(input_ids.shape) > 1 else input_ids
        
        if len(actual_tokens) == probs.shape[0]:
            for i in range(len(actual_tokens)):
                actual_token = actual_tokens[i].item() if hasattr(actual_tokens[i], 'item') else actual_tokens[i]
                token_prob = probs[i, actual_token].item()
                
                # If the actual token has very low probability but neighboring tokens have high prob,
                # this suggests memorization (unnatural probability jumps)
                top5_probs, _ = torch.topk(probs[i], 5)
                avg_top5 = top5_probs.mean().item()
                
                if token_prob < 0.1 * avg_top5:  # Token probability much lower than top predictions
                    token_consistency_penalty += (avg_top5 - token_prob)
    
    # 3. Entropy progression analysis
    # Memorized sequences often show specific entropy progression patterns
    entropies = []
    for i in range(probs.shape[0]):
        entropy = compute_shannon_entropy(probs[i:i+1]).item()
        entropies.append(entropy)
    
    entropy_progression_penalty = 0.0
    if len(entropies) > 1:
        # Sudden entropy drops can indicate memorization
        entropy_diffs = np.diff(entropies)
        # Penalize large negative changes (sudden confidence increases)
        negative_jumps = [diff for diff in entropy_diffs if diff < -1.0]
        if negative_jumps:
            entropy_progression_penalty = sum(abs(jump) for jump in negative_jumps)
    
    # Combine penalties
    coherence_penalty = (
        0.4 * smoothness_penalty +           # Weight smoothness most
        0.3 * token_consistency_penalty +    # Weight token consistency moderately
        0.3 * entropy_progression_penalty    # Weight entropy progression moderately
    )
    
    return coherence_penalty

def multilayer_concentration_fusion(mink_plus_score, layer_features_list, layer_weights, coherence_penalty=0.0):
    """Enhanced fusion with weighted harmonic mean and sequence coherence penalty"""
    if not layer_features_list:
        # Apply coherence penalty to original mink++ score
        penalty_factor = 1.0 + 0.1 * coherence_penalty  # Small penalty adjustment
        return mink_plus_score * penalty_factor
    
    # Enhanced aggregation using weighted harmonic mean for robustness
    aggregated_features = {}
    feature_names = layer_features_list[0].keys()
    
    for feature_name in feature_names:
        feature_values = []
        feature_weights = []
        
        for i, layer_features in enumerate(layer_features_list):
            if feature_name in layer_features and i < len(layer_weights):
                feature_values.append(layer_features[feature_name])
                feature_weights.append(layer_weights[i])
        
        if feature_values:
            # Use weighted harmonic mean for better outlier robustness
            aggregated_features[feature_name] = compute_weighted_harmonic_mean(feature_values, feature_weights)
        else:
            aggregated_features[feature_name] = 0.0
    
    # Normalize features to [-1, 1] range
    normalized_features = normalize_concentration_features(aggregated_features)
    
    # Combine normalized features into a single concentration score
    concentration_score = compute_concentration_score(normalized_features)
    
    # Adaptive fusion with Min-K%++ score
    # Higher concentration (memorized data) should correlate with lower Min-K%++ scores
    alpha = 0.6  # Base weight for Min-K%++
    
    # Adjust alpha based on concentration patterns and coherence
    concentration_strength = abs(concentration_score)
    coherence_factor = min(coherence_penalty / 2.0, 0.2)  # Cap coherence influence
    
    # Higher coherence penalty suggests memorization, so increase weight of concentration features
    alpha_adj = alpha + 0.1 * (concentration_strength - 0.5) - coherence_factor
    alpha_adj = np.clip(alpha_adj, 0.3, 0.8)  # Keep within reasonable bounds
    
    # Final combination with coherence penalty integration
    combined_score = alpha_adj * mink_plus_score + (1 - alpha_adj) * concentration_score
    
    # Apply coherence penalty as a multiplicative factor
    penalty_factor = 1.0 + 0.05 * coherence_penalty  # Small penalty adjustment
    combined_score = combined_score * penalty_factor
    
    return combined_score

def normalize_concentration_features(features):
    """Normalize concentration features to [-1, 1] range"""
    normalized = {}
    
    # Feature normalization ranges based on expected distributions
    ranges = {
        'entropy': (0.0, 10.0),      # Typical entropy range
        'gini': (0.0, 1.0),          # Gini coefficient range
        'top5': (0.0, 1.0),          # Concentration ranges
        'top10': (0.0, 1.0),
        'top20': (0.0, 1.0),
        'top50': (0.0, 1.0),
        'eff_vocab': (0.0, 1.0)      # Normalized effective vocab size
    }
    
    for feature_name, value in features.items():
        if feature_name in ranges:
            min_val, max_val = ranges[feature_name]
            # Clip and normalize to [-1, 1]
            clipped = np.clip(value, min_val, max_val)
            normalized[feature_name] = 2 * (clipped - min_val) / (max_val - min_val) - 1
        else:
            # Default normalization for unknown features
            normalized[feature_name] = np.tanh(value)
    
    return normalized

def compute_concentration_score(normalized_features):
    """Combine normalized concentration features into a single score"""
    # Weights for different features based on their discriminative power
    feature_weights = {
        'entropy': -0.25,        # Lower entropy = more concentrated = higher memorization
        'gini': 0.20,           # Higher gini = more concentrated = higher memorization  
        'top5': 0.15,           # Higher top-k = more concentrated = higher memorization
        'top10': 0.15,
        'top20': 0.10,
        'top50': 0.05,
        'eff_vocab': -0.10      # Lower effective vocab = more concentrated = higher memorization
    }
    
    score = 0.0
    total_weight = 0.0
    
    for feature_name, feature_value in normalized_features.items():
        if feature_name in feature_weights:
            weight = feature_weights[feature_name]
            score += weight * feature_value
            total_weight += abs(weight)
    
    # Normalize by total absolute weight
    if total_weight > 0:
        score = score / total_weight
    
    return score

# Config class
class Config:
    def __init__(self):
        self.models = ['EleutherAI/pythia-2.8b', 'state-spaces/mamba-1.4b-hf']
        self.datasets = ['WikiMIA_length32', 'WikiMIA_length64', 'WikiMIA_length128']
        self.half = False
        self.int8 = False
        self.method = 'mink++_multilayer_concentration'  # Multi-layer concentration method
        self.ratio = 0.6  # Base ratio for Min-K%++
    
    def get_dataset_choices(self):
        return [
            'WikiMIA_length32', 'WikiMIA_length64', 'WikiMIA_length128', 
            'WikiMIA_length32_paraphrased',
            'WikiMIA_length64_paraphrased',
            'WikiMIA_length128_paraphrased', 
        ]

# arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='results', help='output directory')
    return parser.parse_args()

# load model
def load_model(name, args):
    int8_kwargs = {}
    half_kwargs = {}
    if args.int8:
        int8_kwargs = dict(load_in_8bit=True, torch_dtype=torch.bfloat16)
    elif args.half:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
    
    if 'mamba' in name:
        try:
            from transformers import MambaForCausalLM
        except ImportError:
            raise ImportError
        model = MambaForCausalLM.from_pretrained(
            name, return_dict=True, device_map='auto', **int8_kwargs, **half_kwargs
        )        
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name, return_dict=True, device_map='auto', **int8_kwargs, **half_kwargs
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer

# compute metrics
def get_metrics(scores, labels):
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]
    tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]
    return auroc, fpr95, tpr05

def process_dataset(model, tokenizer, dataset_name, config):
    """Process individual dataset with Multi-Layer Concentration Analysis"""
    print(f"\nProcessing dataset: {dataset_name}")
    
    # load dataset
    if not 'paraphrased' in dataset_name:
        dataset = load_dataset('swj0419/WikiMIA', split=dataset_name)
    else:
        dataset = load_dataset('zjysteven/WikiMIA_paraphrased_perturbed', split=dataset_name)
    data = convert_huggingface_data_to_list_dic(dataset)
    
    # Determine layer indices for multi-layer analysis
    num_layers, _ = get_model_layers(model)
    if num_layers > 0:
        # Select layers at 1/4, 1/2, and 3/4 depth
        layer_indices = [
            max(0, num_layers // 4),
            max(0, num_layers // 2),
            max(0, 3 * num_layers // 4)
        ]
        layer_weights = [0.3, 0.4, 0.3]  # Weights for early, middle, late layers
        print(f"Model has {num_layers} layers. Using layers: {layer_indices}")
    else:
        layer_indices = []
        layer_weights = []
        print("Multi-layer analysis not available for this model architecture.")
    
    # inference - get scores for each input
    scores = defaultdict(list)
    for i, d in enumerate(tqdm(data, total=len(data), desc=f'Samples ({dataset_name})')): 
        text = d['input']
        
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
        input_ids = input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        ll = -loss.item() # log-likelihood

        # mink and mink++
        input_ids = input_ids[0][1:].unsqueeze(-1)
        probs = F.softmax(logits[0, :-1], dim=-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

        # Original mink++
        if config.method in ['mink++', 'both', 'mink++_multilayer_concentration']:
            mink_plus = (token_log_probs - mu) / sigma.sqrt()
            k_length = int(len(mink_plus) * config.ratio)
            topk = np.sort(mink_plus.cpu())[:k_length]
            mink_plus_score = np.mean(topk).item()
            
            if config.method == 'mink++':
                scores[f'mink++_{config.ratio}'].append(mink_plus_score)
            
            # Enhanced method with Multi-Layer Concentration Analysis
            if config.method == 'mink++_multilayer_concentration':
                # Extract features from intermediate layers
                layer_features_list = []
                
                if layer_indices:
                    # Re-tokenize for layer extraction (needed for hooks)
                    layer_input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(model.device)
                    layer_outputs = extract_intermediate_layers(model, layer_input_ids, layer_indices)
                    
                    if layer_outputs:
                        for layer_idx in layer_indices:
                            if layer_idx in layer_outputs:
                                layer_probs = layer_outputs[layer_idx]
                                layer_features = compute_layer_concentration_features(layer_probs, layer_idx)
                                layer_features_list.append(layer_features)
                
                # Compute sequence coherence penalty
                coherence_penalty = 0.0
                try:
                    # Use the output probabilities from the final layer for coherence analysis
                    coherence_penalty = compute_sequence_coherence_penalty(probs, input_ids.cpu())
                except Exception as e:
                    # Fallback if coherence computation fails
                    print(f"Warning: Coherence penalty computation failed: {e}")
                    coherence_penalty = 0.0
                
                # Combine with multi-layer concentration analysis and coherence penalty
                combined_score = multilayer_concentration_fusion(
                    mink_plus_score, layer_features_list, layer_weights, coherence_penalty
                )
                scores[f'mink++_multilayer_concentration_{config.ratio}'].append(combined_score)

    # compute metrics
    labels = [d['label'] for d in data] # 1: training, 0: non-training

    results = defaultdict(list)
    for method, method_scores in scores.items():
        auroc, fpr95, tpr05 = get_metrics(method_scores, labels)
        
        results['method'].append(method)
        results['auroc'].append(f"{auroc:.1%}")
        results['fpr95'].append(f"{fpr95:.1%}")
        results['tpr05'].append(f"{tpr05:.1%}")

    return results, scores, labels


def get_best_method(results):
    # Collect all AUROC values for each method across all models and datasets
    method_auroc_scores = {}  # method_name -> list of auroc scores
    
    def to_float(x):
            return float(str(x).rstrip('%')) / 100 if isinstance(x, str) else float(x)
    
    for model_name, model_data in results.items():
        for dataset_name, dataset_data in model_data.items():
            methods = dataset_data.get('method', [])
            auroc_values = dataset_data.get('auroc', [])
            
            for method, auroc_value in zip(methods, auroc_values):
                if method not in method_auroc_scores:
                    method_auroc_scores[method] = []
                method_auroc_scores[method].append(to_float(auroc_value))
    
    # Check if we have any methods at all
    if not method_auroc_scores:
        print("Warning: No methods found in results. Using default method.")
        return 'mink++_multilayer_concentration_0.6'  # Default fallback
    
    # Calculate mean AUROC for each method and find the best one
    method_mean_auroc = {}
    for method, scores in method_auroc_scores.items():
        method_mean_auroc[method] = sum(scores) / len(scores)
    
    # Find the method with highest mean AUROC
    sorted_methods = sorted(method_mean_auroc.items(), key=lambda x: x[1], reverse=True)
    
    # Check if sorted_methods is empty
    if not sorted_methods:
        print("Warning: No methods available after sorting. Using default method.")
        return 'mink++_multilayer_concentration_0.6'  # Default fallback
    
    best_method = sorted_methods[0][0]
    
    # For our improved method, we want to compare against the baseline
    # Don't exclude our proposed method
    if re.match(r"^mink\+\+_multilayer_concentration_\d+(\.\d+)?$", best_method):
        # Keep the improved method as best if it's truly best
        pass
    elif re.match(r"^mink\+\+_\d+(\.\d+)?$", best_method):
        if len(sorted_methods) > 1:
            best_method = sorted_methods[1][0]
            print(f"Best method matched mink++ pattern, using second best: {best_method}")
        else:
            print(f"Only one method available, keeping: {best_method}")
    
    return best_method


def save_best_method_results(all_results, all_scores, best_method, save_root):
    """Save results and scores for the best method"""
    # Extract best method results only
    best_method_results = extract_best_method_results(all_results, best_method)
    
    # Save results to JSON file
    best_method_results_file = os.path.join(save_root, "best_method_results.json")
    with open(best_method_results_file, 'w', encoding='utf-8') as f:
        json.dump(best_method_results, f, indent=2, ensure_ascii=False)
    
    # Extract best method scores only
    best_method_scores = extract_best_method_scores(all_scores, best_method)
    
    # Save scores to pickle file
    best_method_scores_file = os.path.join(save_root, "scores.pkl")
    with open(best_method_scores_file, 'wb') as f:
        pickle.dump(best_method_scores, f)
    
    return best_method_results_file, best_method_scores_file


def extract_best_method_results(all_results, best_method):
    """Extract results for the best method only"""
    best_method_results = {}
    for model_name, model_data in all_results.items():
        if model_name not in best_method_results:
            best_method_results[model_name] = {}
        for dataset_name, dataset_data in model_data.items():
            print("dataset_data[method]:", dataset_data["method"])
            if best_method in dataset_data["method"]:
                # Get index of best method
                method_index = dataset_data["method"].index(best_method)
                best_method_results[model_name][dataset_name] = {
                    'method': [dataset_data["method"][method_index]],
                    'auroc': [dataset_data["auroc"][method_index]],
                    'fpr95': [dataset_data["fpr95"][method_index]],
                    'tpr05': [dataset_data["tpr05"][method_index]]
                }
    return best_method_results


def extract_best_method_scores(all_scores, best_method):
    """Extract scores for the best method only"""
    best_method_scores = {}
    for model_name, model_data in all_scores.items():
        if model_name not in best_method_scores:
            best_method_scores[model_name] = {}
        for dataset_name, dataset_data in model_data.items():
            if best_method in dataset_data:
                best_method_scores[model_name][dataset_name] = {
                    best_method: dataset_data[best_method]
                }
    return best_method_scores


def main():
    args = parse_arguments()
    config = Config()
    
    all_results = {}
    all_scores = {}
        
    # load model
    for model_name in config.models:
        model, tokenizer = load_model(model_name, config)
        model_id = model_name.split('/')[-1]
        all_results[model_id] = {}
        all_scores[model_id] = {}
        
        # Process each dataset sequentially
        for dataset_name in config.datasets:
            try:
                results, scores, labels = process_dataset(model, tokenizer, dataset_name, config)
                
                all_results[model_id][dataset_name] = {
                    'method': results['method'],
                    'auroc': results['auroc'],
                    'fpr95': results['fpr95'],
                    'tpr05': results['tpr05']
                }
                
                all_scores[model_id][dataset_name] = {}
                for method, method_scores in scores.items():
                    score_dict = {"training": [], "non-training": []}
                    for label, score in zip(labels, method_scores):
                        if label == 1:
                            score_dict["training"].append(score)
                        elif label == 0:
                            score_dict["non-training"].append(score)
                        else:
                            raise ValueError(f"Invalid label: {label}")
                    all_scores[model_id][dataset_name][method] = score_dict
                
                df = pd.DataFrame(results)
                print(f"\nResults for {dataset_name}:")
                print(df)
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                continue

        print("\n" + "="*50)
        print("RESULTS SUMMARY:")
        print("="*50)
        
        for dataset_name, dataset_results in all_results[model_id].items():
            print(f"\n{dataset_name}:")
            df = pd.DataFrame(dataset_results)
            print(df)

    # save results
    save_root = os.path.join(args.output_dir)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    result_file = os.path.join(save_root, "results.json")
    
    # Save to JSON file
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # best method
    best_method = get_best_method(all_results)
    print(f"Best method: {best_method}")
    with open(os.path.join(save_root, "best_method.txt"), 'w', encoding='utf-8') as f:
        f.write(best_method)

    # save best method results and scores
    best_method_results_file, best_method_scores_file = save_best_method_results(all_results, all_scores, best_method, save_root)
    
    print(f"\nResults saved to {result_file}")
    print(f"Best method results saved to {best_method_results_file}")
    print(f"Best method scores saved to {best_method_scores_file}")

if __name__ == "__main__":
    main()