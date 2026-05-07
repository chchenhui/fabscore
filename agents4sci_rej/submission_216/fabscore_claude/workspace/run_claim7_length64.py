"""
Minimal script to run Min-K%++ baseline for Pythia-2.8b on WikiMIA_length64.
Verifies claim 7: AUROC 63.8% for Pythia-2.8b / Length 64 / Min-K%++
"""
import sys
sys.path.insert(0, '/home/chenhui/fabscore/agents4sci_rej/submission_216/Mink%++')

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json

def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data

def main():
    model_name = 'EleutherAI/pythia-2.8b'
    dataset_name = 'WikiMIA_length64'
    ratio = 0.6

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to('cuda:0')
    model.eval()

    print(f"Loading dataset: swj0419/WikiMIA split={dataset_name}")
    dataset = load_dataset('swj0419/WikiMIA', split=dataset_name)
    data = convert_huggingface_data_to_list_dic(dataset)
    print(f"Dataset size: {len(data)}")

    scores = []
    labels = []

    print("Computing Min-K%++ scores...")
    for i, d in enumerate(tqdm(data)):
        text = d['input']
        label = d['label']
        labels.append(label)

        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

        input_ids_shifted = input_ids[0][1:].unsqueeze(-1)
        probs = torch.softmax(logits[0, :-1], dim=-1)
        log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=input_ids_shifted).squeeze(-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

        mink_plus = (token_log_probs - mu) / sigma.sqrt()
        k_length = int(len(mink_plus) * ratio)
        topk = np.sort(mink_plus.cpu().numpy())[:k_length]
        mink_plus_score = np.mean(topk).item()
        scores.append(mink_plus_score)

    fpr_list, tpr_list, _ = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]
    tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]

    print(f"\n=== Min-K%++ Results for {model_name} on {dataset_name} ===")
    print(f"AUROC: {auroc*100:.1f}%  (paper claims: 63.8%)")
    print(f"FPR95: {fpr95*100:.1f}%")
    print(f"TPR05: {tpr05*100:.1f}%")

    result = {
        'model': model_name,
        'dataset': dataset_name,
        'method': f'mink++_{ratio}',
        'auroc': f'{auroc*100:.4f}%',
        'auroc_raw': auroc,
        'fpr95': f'{fpr95*100:.4f}%',
        'fpr95_raw': float(fpr95),
        'tpr05': f'{tpr05*100:.4f}%',
        'tpr05_raw': float(tpr05),
    }
    out_path = '/home/chenhui/fabscore/agents4sci_rej/submission_216/fabscore_claude/workspace/claim7_length64_result.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
