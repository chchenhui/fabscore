"""
Compute FPR95 for Min-K%++ baseline for Pythia-2.8b on WikiMIA_length32.
Verifies claim 2: FPR95 87.1% for Pythia-2.8b / Length 32 / Min-K%++
"""
import sys
import os
sys.path.insert(0, '/home/chenhui/fabscore/agents4sci_rej/submission_216/Mink%++')

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json

def compute_fpr_at_tpr(fpr, tpr, target_tpr=0.95):
    """Compute FPR at a given TPR threshold."""
    # Find the index where TPR >= target_tpr
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return 1.0
    return fpr[idx[0]]

def main():
    model_name = 'EleutherAI/pythia-2.8b'
    dataset_name = 'WikiMIA_length32'
    ratio = 0.6

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    model = model.to('cuda:0')
    model.eval()

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset('swj0419/WikiMIA', split=dataset_name)
    data = list(dataset)

    scores = []
    labels = []

    print("Computing Min-K%++ scores...")
    for d in tqdm(data):
        text = d['input']
        label = d['label']
        labels.append(label)

        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(model.device)

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

    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)
    fpr95 = compute_fpr_at_tpr(fpr, tpr, target_tpr=0.95)

    print(f"\n=== Min-K%++ Results for {model_name} on {dataset_name} ===")
    print(f"AUROC: {auroc*100:.1f}%")
    print(f"FPR95: {fpr95*100:.1f}%")
    print(f"Paper claim 1 (AUROC): 64.4%")
    print(f"Paper claim 2 (FPR95): 87.1%")

    result = {
        'model': model_name,
        'dataset': dataset_name,
        'method': f'mink++_{ratio}',
        'auroc': f'{auroc*100:.1f}%',
        'auroc_raw': auroc,
        'fpr95': f'{fpr95*100:.1f}%',
        'fpr95_raw': fpr95
    }
    out_path = '/home/chenhui/fabscore/agents4sci_rej/submission_216/fabscore_claude/workspace/claim2_fpr95_result.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
