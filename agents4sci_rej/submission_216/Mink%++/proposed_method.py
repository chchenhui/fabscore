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

# Distribution shape analysis functions
def compute_skewness(probs):
    """Compute skewness of probability distributions"""
    # Convert to numpy for scipy stats
    probs_np = probs.detach().cpu().numpy()
    skewness_values = []
    
    for prob_dist in probs_np:
        # Avoid zero probabilities for numerical stability
        prob_dist = np.clip(prob_dist, 1e-10, 1.0)
        prob_dist = prob_dist / prob_dist.sum()  # Renormalize
        
        # Create discrete distribution
        indices = np.arange(len(prob_dist))
        mean = np.sum(indices * prob_dist)
        variance = np.sum(((indices - mean) ** 2) * prob_dist)
        
        if variance > 1e-10:
            skew = np.sum(((indices - mean) ** 3) * prob_dist) / (variance ** 1.5)
        else:
            skew = 0.0
        
        skewness_values.append(skew)
    
    return np.array(skewness_values)

def compute_kurtosis(probs):
    """Compute kurtosis of probability distributions"""
    probs_np = probs.detach().cpu().numpy()
    kurtosis_values = []
    
    for prob_dist in probs_np:
        # Avoid zero probabilities for numerical stability
        prob_dist = np.clip(prob_dist, 1e-10, 1.0)
        prob_dist = prob_dist / prob_dist.sum()  # Renormalize
        
        # Create discrete distribution
        indices = np.arange(len(prob_dist))
        mean = np.sum(indices * prob_dist)
        variance = np.sum(((indices - mean) ** 2) * prob_dist)
        
        if variance > 1e-10:
            kurt = np.sum(((indices - mean) ** 4) * prob_dist) / (variance ** 2) - 3.0
        else:
            kurt = 0.0
            
        kurtosis_values.append(kurt)
    
    return np.array(kurtosis_values)

def compute_entropy(probs):
    """Compute entropy of probability distributions"""
    # Add small epsilon to avoid log(0)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.detach().cpu().numpy()

def compute_distribution_shape_features(probs):
    """Extract all distribution shape features"""
    skewness = compute_skewness(probs)
    kurtosis = compute_kurtosis(probs)
    entropy = compute_entropy(probs)
    
    return {
        'skewness': np.mean(skewness),
        'kurtosis': np.mean(kurtosis),
        'entropy': np.mean(entropy)
    }

def normalize_features(features_dict):
    """Normalize shape features to standard scale"""
    # Simple min-max normalization to [-1, 1] range
    normalized = {}
    
    # Typical ranges for normalization (based on empirical observations)
    feature_ranges = {
        'skewness': (-2.0, 2.0),
        'kurtosis': (-2.0, 10.0),
        'entropy': (0.0, 10.0)
    }
    
    for key, value in features_dict.items():
        min_val, max_val = feature_ranges.get(key, (-1, 1))
        # Clip and normalize to [-1, 1]
        clipped = np.clip(value, min_val, max_val)
        normalized[key] = 2 * (clipped - min_val) / (max_val - min_val) - 1
    
    return normalized

def combine_scores(mink_plus_score, shape_features, alpha=0.7):
    """Combine Min-K%++ score with distribution shape features"""
    normalized_features = normalize_features(shape_features)
    
    # Take weighted average of normalized features
    shape_score = np.mean(list(normalized_features.values()))
    
    # Combine with Min-K%++ score
    combined_score = alpha * mink_plus_score + (1 - alpha) * shape_score
    return combined_score

# Config class
class Config:
    def __init__(self):
        self.models = ['EleutherAI/pythia-2.8b', 'state-spaces/mamba-1.4b-hf']
        self.datasets = ['WikiMIA_length32', 'WikiMIA_length64', 'WikiMIA_length128']
        self.half = False
        self.int8 = False
        self.method = 'mink++_shape' # New method with shape analysis
        self.ratio = 0.6
    
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
    """Process individual dataset with enhanced shape analysis"""
    print(f"\nProcessing dataset: {dataset_name}")
    
    # load dataset
    if not 'paraphrased' in dataset_name:
        dataset = load_dataset('swj0419/WikiMIA', split=dataset_name)
    else:
        dataset = load_dataset('zjysteven/WikiMIA_paraphrased_perturbed', split=dataset_name)
    data = convert_huggingface_data_to_list_dic(dataset)
    
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

        # Get probabilities and log probabilities
        input_ids = input_ids[0][1:].unsqueeze(-1)
        probs = F.softmax(logits[0, :-1], dim=-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

        # Original mink++
        if config.method in ['mink++', 'both', 'mink++_shape']:
            mink_plus = (token_log_probs - mu) / sigma.sqrt()
            k_length = int(len(mink_plus) * config.ratio)
            topk = np.sort(mink_plus.cpu())[:k_length]
            mink_plus_score = np.mean(topk).item()
            
            if config.method == 'mink++':
                scores[f'mink++_{config.ratio}'].append(mink_plus_score)
            
            # Enhanced method with distribution shape analysis
            if config.method == 'mink++_shape':
                # Extract distribution shape features
                shape_features = compute_distribution_shape_features(probs)
                
                # Combine with Min-K%++ score
                combined_score = combine_scores(mink_plus_score, shape_features)
                scores[f'mink++_shape_{config.ratio}'].append(combined_score)

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
    
    # Calculate mean AUROC for each method and find the best one
    method_mean_auroc = {}
    for method, scores in method_auroc_scores.items():
        method_mean_auroc[method] = sum(scores) / len(scores)
    
    # Find the method with highest mean AUROC
    sorted_methods = sorted(method_mean_auroc.items(), key=lambda x: x[1], reverse=True)
    best_method = sorted_methods[0][0]
    
    # For our new method, we want to compare against the baseline
    # Don't exclude our proposed method
    if re.match(r"^mink\+\+_shape_\d+(\.\d+)?$", best_method):
        # Keep the proposed method as best if it's truly best
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