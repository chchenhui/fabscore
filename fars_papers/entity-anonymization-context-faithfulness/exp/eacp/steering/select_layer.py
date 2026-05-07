"""Layer selection for ContextFocus. Evaluates steering at each layer on
200 held-out NQ-SWAP examples, selecting the layer that maximizes Ps
(context-aligned answer rate) improvement. Based on Anand et al. 2026."""

import argparse
import json
import os
import re
import string
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
VECTORS_DIR = os.path.join(SCRIPT_DIR, "vectors")

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = 8
MULTIPLIER = 2.0


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def compute_ps(predictions, examples):
    """Compute Ps = fraction where normalized sub_answer is recalled in output."""
    hits = 0
    for pred, ex in zip(predictions, examples):
        sub_ans = ex["sub_answer"]
        if isinstance(sub_ans, list):
            sub_ans = sub_ans[0] if sub_ans else ""
        if normalize_answer(sub_ans) in normalize_answer(pred):
            hits += 1
    return hits / len(predictions) if predictions else 0.0


class SteeringHook:
    def __init__(self, steering_vector, multiplier, input_lengths):
        self.steering_vector = steering_vector
        self.multiplier = multiplier
        self.input_lengths = input_lengths

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        if self.multiplier == 0:
            return output

        device = hidden.device
        sv = self.steering_vector.to(device=device, dtype=hidden.dtype)
        seq_len = hidden.size(1)

        if seq_len == 1:
            hidden[:, :, :] = hidden[:, :, :] + self.multiplier * sv
        else:
            for b in range(hidden.size(0)):
                if b < len(self.input_lengths):
                    start_pos = self.input_lengths[b]
                    if start_pos < seq_len:
                        hidden[b, start_pos:] = hidden[b, start_pos:] + self.multiplier * sv

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden


def generate_with_steering(model, tokenizer, texts, steering_vector, layer_idx,
                           multiplier, device, batch_size=BATCH_SIZE, max_new_tokens=32):
    """Generate with optional steering at a specific layer."""
    all_outputs = []

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

        hook_obj = SteeringHook(steering_vector, multiplier, input_lengths)
        handle = model.model.layers[layer_idx].register_forward_hook(hook_obj)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        handle.remove()

        for i, (inp_ids, out_ids) in enumerate(zip(inputs["input_ids"], output_ids)):
            inp_len = input_lengths[i]
            gen_ids = out_ids[inp_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            all_outputs.append(text.strip())

    return all_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--vectors_path", default=os.path.join(VECTORS_DIR, "llama31_8b_contextfocus.pt"))
    parser.add_argument("--data_path", default=os.path.join(DATA_DIR, "nqswap_layerselect.json"))
    args = parser.parse_args()

    with open(args.data_path) as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} layer-selection examples")

    steering_vectors = torch.load(args.vectors_path, weights_only=True)
    num_layers = steering_vectors.shape[0]
    print(f"Loaded steering vectors: {steering_vectors.shape}")

    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    pos_texts = [ex["positive_text"] for ex in examples]

    print("\nComputing baseline Ps (m=0)...")
    t0 = time.time()
    baseline_preds = generate_with_steering(
        model, tokenizer, pos_texts, steering_vectors[0], 0, 0.0, device, args.batch_size
    )
    baseline_ps = compute_ps(baseline_preds, examples)
    print(f"Baseline Ps = {baseline_ps:.4f} ({baseline_ps*100:.2f}%), took {time.time()-t0:.1f}s")

    print(f"\nEvaluating steering at each of {num_layers} layers with m={MULTIPLIER}...")
    layer_results = []
    best_layer = -1
    best_improvement = -float("inf")

    for l in range(num_layers):
        t0 = time.time()
        preds = generate_with_steering(
            model, tokenizer, pos_texts, steering_vectors[l], l,
            MULTIPLIER, device, args.batch_size
        )
        ps = compute_ps(preds, examples)
        improvement = ps - baseline_ps
        elapsed = time.time() - t0

        layer_results.append({
            "layer": l,
            "ps_steered": round(ps, 4),
            "ps_baseline": round(baseline_ps, 4),
            "improvement": round(improvement, 4),
        })

        if improvement > best_improvement:
            best_improvement = improvement
            best_layer = l

        print(f"  Layer {l:2d}: Ps = {ps*100:.2f}% (delta = {improvement*100:+.2f}pp), {elapsed:.1f}s")

    print(f"\nBest layer: {best_layer} (improvement = {best_improvement*100:+.2f}pp)")
    print(f"Best Ps = {layer_results[best_layer]['ps_steered']*100:.2f}%")

    result = {
        "selected_layer": best_layer,
        "baseline_ps": round(baseline_ps, 4),
        "best_ps": layer_results[best_layer]["ps_steered"],
        "improvement": round(best_improvement, 4),
        "multiplier": MULTIPLIER,
        "num_examples": len(examples),
        "all_layers": layer_results,
    }

    os.makedirs(VECTORS_DIR, exist_ok=True)
    out_path = os.path.join(VECTORS_DIR, "selected_layer.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved layer selection result to {out_path}")


if __name__ == "__main__":
    main()
