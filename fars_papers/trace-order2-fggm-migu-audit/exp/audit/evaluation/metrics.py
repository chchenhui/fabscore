# OP (Overall Performance), BWT (Backward Transfer), and per-task metric computation.
# Ported from external/TRACE/metrics.py and external/TRACE/evaluations/.
# C-STANCE and FOMC both use accuracy (matching TRACE reference code, not the task
# description text which mentions F1-macro/F1).

import re
import numpy as np
from typing import List, Dict, Optional
from rouge import Rouge
from fuzzywuzzy import fuzz
from nltk.translate.bleu_score import sentence_bleu


TASK_PRIMARY_METRIC = {
    "C-STANCE": "accuracy",
    "FOMC": "accuracy",
    "MeetingBank": "rouge-L",
    "Py150": "similarity",
    "ScienceQA": "accuracy",
    "NumGLUE-cm": "accuracy",
    "NumGLUE-ds": "accuracy",
    "20Minuten": "sari",
}


def tokenize_bleu(text):
    tokens = re.split(r'\s|\.', text)
    return [t for t in tokens if len(t) > 0]


def bleu_score(reference, hypothesis, gram):
    ref_tokens = tokenize_bleu(reference)
    hyp_tokens = tokenize_bleu(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0
    weights = {
        1: (1.,),
        2: (1./2., 1./2.),
        3: (1./3., 1./3., 1./3.),
        4: (1./4., 1./4., 1./4., 1./4.),
    }
    try:
        return sentence_bleu([ref_tokens], hyp_tokens, weights[gram])
    except Exception:
        return 0.0


def calculate_bleu(predictions, references, gram):
    bleus = []
    for pred, ref in zip(predictions, references):
        if pred == "" or ref == "":
            continue
        bleus.append(bleu_score(ref, pred, gram))
    return sum(bleus) / len(predictions) if predictions else 0.0


def calculate_rouge(predictions, references):
    rouge = Rouge(metrics=["rouge-l"])
    rouges = []
    for pred, ref in zip(predictions, references):
        if pred == "" or ref == "":
            continue
        try:
            scores = rouge.get_scores(pred, ref, avg=True)
            rouges.append(scores['rouge-l']['f'])
        except Exception:
            continue
    return sum(rouges) / len(predictions) if predictions else 0.0


def calculate_accuracy(predictions, references):
    scores = 0
    for pred, ref in zip(predictions, references):
        if pred == "" or ref == "":
            continue
        if pred.strip() == ref.strip():
            scores += 1
    return scores / len(predictions) if predictions else 0.0


def calculate_fuzz(predictions, references):
    scores = 0
    for pred, ref in zip(predictions, references):
        if pred == "" or ref == "":
            continue
        scores += fuzz.ratio(pred, ref)
    return scores / len(predictions) if predictions else 0.0


def calculate_sari(sources, predictions, references):
    try:
        import evaluate
        sari_metric = evaluate.load("sari")
        result = sari_metric.compute(
            sources=sources,
            predictions=predictions,
            references=[[r] for r in references],
        )
        return result["sari"]
    except Exception:
        try:
            from easse.sari import corpus_sari
            return corpus_sari(
                orig_sents=sources,
                sys_sents=predictions,
                refs_sents=[references],
            )
        except Exception:
            return 0.0


def postprocess_py150(code):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code


def resolve_scienceqa(dataset):
    answers = []
    reasonings = []
    for item in dataset:
        if len(item) > 0:
            answers.append(item[0])
            reasonings.append(item[2:] if len(item) > 2 else "")
        else:
            answers.append("")
            reasonings.append("")
    return answers, reasonings


def compute_task_metric(
    task_name: str,
    predictions: List[str],
    references: List[str],
    sources: Optional[List[str]] = None,
) -> Dict:
    predictions = [p.strip() for p in predictions]
    references = [r.strip() for r in references]

    if task_name == "C-STANCE":
        preds_first = [p[0] if p else "" for p in predictions]
        refs_first = [r[0] if r else "" for r in references]
        acc = calculate_accuracy(preds_first, refs_first) * 100
        return {"accuracy": acc, "primary": acc}

    elif task_name == "FOMC":
        preds_first = [p[0] if p else "" for p in predictions]
        refs_first = [r[0] if r else "" for r in references]
        acc = calculate_accuracy(preds_first, refs_first) * 100
        return {"accuracy": acc, "primary": acc}

    elif task_name == "MeetingBank":
        bleu_1 = calculate_bleu(predictions, references, 1) * 100
        bleu_4 = calculate_bleu(predictions, references, 4) * 100
        rouge_l = calculate_rouge(predictions, references) * 100
        return {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge_l, "primary": rouge_l}

    elif task_name == "Py150":
        preds_pp = [postprocess_py150(p) for p in predictions]
        refs_pp = [postprocess_py150(r) for r in references]
        sim = calculate_fuzz(preds_pp, refs_pp)
        return {"similarity": sim, "primary": sim}

    elif task_name == "ScienceQA":
        pred_answers, pred_reasonings = resolve_scienceqa(predictions)
        ref_answers, ref_reasonings = resolve_scienceqa(references)
        acc = calculate_accuracy(pred_answers, ref_answers) * 100
        bleu_1 = calculate_bleu(pred_reasonings, ref_reasonings, 1) * 100
        bleu_4 = calculate_bleu(pred_reasonings, ref_reasonings, 4) * 100
        rouge_l = calculate_rouge(pred_reasonings, ref_reasonings) * 100
        return {"accuracy": acc, "bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge_l, "primary": acc}

    elif task_name == "NumGLUE-cm":
        preds_line = [p.split("\n")[0].strip() for p in predictions]
        acc = calculate_accuracy(preds_line, references) * 100
        return {"accuracy": acc, "primary": acc}

    elif task_name == "NumGLUE-ds":
        preds_line = [p.split("\n")[0].strip() for p in predictions]
        acc = calculate_accuracy(preds_line, references) * 100
        return {"accuracy": acc, "primary": acc}

    elif task_name == "20Minuten":
        bleu_1 = calculate_bleu(predictions, references, 1) * 100
        bleu_4 = calculate_bleu(predictions, references, 4) * 100
        rouge_l = calculate_rouge(predictions, references) * 100
        sari = calculate_sari(sources or [], predictions, references) if sources else 0.0
        return {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge_l, "sari": sari, "primary": sari}

    else:
        return {"primary": 0.0}


def compute_op(performance_matrix: Dict[int, Dict[str, float]], T: int) -> float:
    if T == 0:
        return 0.0
    total = 0.0
    for task_name, score in performance_matrix[T - 1].items():
        total += score
    return total / len(performance_matrix[T - 1])


def compute_bwt(performance_matrix: Dict[int, Dict[str, float]], task_order: List[str], T: int) -> float:
    if T <= 1:
        return 0.0
    bwt_sum = 0.0
    count = 0
    for i in range(T - 1):
        task_name = task_order[i]
        final_score = performance_matrix[T - 1].get(task_name, 0.0)
        diag_score = performance_matrix[i].get(task_name, 0.0)
        bwt_sum += (final_score - diag_score)
        count += 1
    return bwt_sum / count if count > 0 else 0.0
