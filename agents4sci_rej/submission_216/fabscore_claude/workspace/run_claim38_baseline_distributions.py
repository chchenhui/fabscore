"""
Verify claim 38: Figure 2 - Baseline Min-K%++ score distributions for both models/lengths.
Generates score distributions for training (blue) and non-training (red) data
for Pythia-2.8b and Mamba-1.4b-hf across all 3 sequence lengths.
"""
import sys
import os
sys.path.insert(0, '/home/chenhui/fabscore/agents4sci_rej/submission_216/Mink%++')

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from datasets import load_dataset
import json
import pickle

WORKSPACE = '/home/chenhui/fabscore/agents4sci_rej/submission_216/fabscore_claude/workspace'

DATASETS = ['WikiMIA_length32', 'WikiMIA_length64', 'WikiMIA_length128']
RATIO = 0.6

def convert_huggingface_data_to_list_dic(dataset):
    return [dataset[i] for i in range(len(dataset))]

def compute_minkpp_scores(model, tokenizer, data, ratio, device):
    """Compute Min-K%++ scores for a list of data samples."""
    scores = []
    labels = []
    for d in tqdm(data):
        text = d['input']
        label = d['label']
        labels.append(label)
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        input_ids_shifted = input_ids[0][1:].unsqueeze(-1)
        probs = F.softmax(logits[0, :-1], dim=-1)
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids_shifted).squeeze(-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
        mink_plus = (token_log_probs - mu) / sigma.sqrt()
        k_length = int(len(mink_plus) * ratio)
        topk = np.sort(mink_plus.cpu().numpy())[:k_length]
        mink_plus_score = np.mean(topk).item()
        scores.append(mink_plus_score)
    return scores, labels

def process_model(model_name, model_key, model_type, tokenizer_class, model_class, all_scores_data):
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name}")
    print(f"{'='*60}")
    device = 'cuda:0'

    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()

    all_scores_data[model_key] = {}
    metrics = {}

    for dataset_name in DATASETS:
        print(f"\nLoading dataset: {dataset_name}")
        dataset = load_dataset('swj0419/WikiMIA', split=dataset_name)
        data = convert_huggingface_data_to_list_dic(dataset)
        print(f"Dataset size: {len(data)}")

        scores, labels = compute_minkpp_scores(model, tokenizer, data, RATIO, device)

        labels_arr = np.array(labels)
        scores_arr = np.array(scores)

        training_scores = scores_arr[labels_arr == 1].tolist()
        non_training_scores = scores_arr[labels_arr == 0].tolist()

        all_scores_data[model_key][dataset_name] = {
            'mink++': {
                'training': training_scores,
                'non-training': non_training_scores
            }
        }

        fpr_list, tpr_list, _ = roc_curve(labels, scores)
        auroc = auc(fpr_list, tpr_list)
        fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]
        tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]

        metrics[dataset_name] = {
            'auroc': float(auroc),
            'fpr95': float(fpr95),
            'tpr05': float(tpr05),
            'n_training': len(training_scores),
            'n_non_training': len(non_training_scores),
            'training_mean': float(np.mean(training_scores)),
            'training_std': float(np.std(training_scores)),
            'non_training_mean': float(np.mean(non_training_scores)),
            'non_training_std': float(np.std(non_training_scores)),
            'separation': float(np.mean(training_scores) - np.mean(non_training_scores))
        }

        print(f"  AUROC: {auroc*100:.4f}%")
        print(f"  FPR95: {fpr95*100:.4f}%")
        print(f"  TPR05: {tpr05*100:.4f}%")
        print(f"  Training mean: {np.mean(training_scores):.4f} ± {np.std(training_scores):.4f}")
        print(f"  Non-training mean: {np.mean(non_training_scores):.4f} ± {np.std(non_training_scores):.4f}")
        print(f"  Separation (train - non-train): {metrics[dataset_name]['separation']:.4f}")

    del model
    torch.cuda.empty_cache()
    return metrics

def main():
    from transformers import AutoTokenizer, GPTNeoXForCausalLM, MambaForCausalLM

    all_scores_data = {}
    all_metrics = {}

    # Process Pythia-2.8b
    pythia_metrics = process_model(
        'EleutherAI/pythia-2.8b',
        'pythia-2.8b',
        'gptneox',
        AutoTokenizer,
        GPTNeoXForCausalLM,
        all_scores_data
    )
    all_metrics['pythia-2.8b'] = pythia_metrics

    # Process Mamba-1.4b-hf
    mamba_metrics = process_model(
        'state-spaces/mamba-1.4b-hf',
        'mamba-1.4b-hf',
        'mamba',
        AutoTokenizer,
        MambaForCausalLM,
        all_scores_data
    )
    all_metrics['mamba-1.4b-hf'] = mamba_metrics

    # Save raw score distributions
    scores_pkl_path = os.path.join(WORKSPACE, 'baseline_scores.pkl')
    with open(scores_pkl_path, 'wb') as f:
        pickle.dump(all_scores_data, f)
    print(f"\nSaved baseline scores to {scores_pkl_path}")

    # Save metrics summary
    metrics_path = os.path.join(WORKSPACE, 'claim38_baseline_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Summary comparison for separability
    print("\n" + "="*60)
    print("SEPARABILITY COMPARISON (claim: Mamba > Pythia)")
    print("="*60)
    for dataset in DATASETS:
        p_auroc = all_metrics['pythia-2.8b'][dataset]['auroc']
        m_auroc = all_metrics['mamba-1.4b-hf'][dataset]['auroc']
        print(f"{dataset}:")
        print(f"  Pythia-2.8b AUROC: {p_auroc*100:.4f}%")
        print(f"  Mamba-1.4b-hf AUROC: {m_auroc*100:.4f}%")
        print(f"  Mamba superior: {m_auroc > p_auroc}")

    print("\nDone!")
    return all_scores_data, all_metrics

if __name__ == '__main__':
    main()
