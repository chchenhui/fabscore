# Evaluation metrics for MMStar and HallusionBench.
# compute_mmstar_accuracy: fraction of correct MC answers.
# compute_hallusionbench_aacc: per-question accuracy (aAcc) following the
# official protocol with regex Yes/No extraction.


def compute_mmstar_accuracy(predictions):
    if not predictions:
        return 0.0
    correct = sum(1 for p in predictions if p["extracted_answer"] == p["gt_answer"])
    return correct / len(predictions)


def compute_hallusionbench_aacc(predictions):
    if not predictions:
        return 0.0

    for p in predictions:
        gt = str(p["gt_answer"])
        pred = str(p["extracted_answer"])

        if p.get("category") == "VS" and str(p.get("figure_id")) == "0":
            p["correct"] = 1 if (pred == gt or pred == "2") else 0
        else:
            p["correct"] = 1 if pred == gt else 0

    total = len(predictions)
    correct = sum(p["correct"] for p in predictions)
    aacc = correct / total if total > 0 else 0.0

    vd_preds = [p for p in predictions if p.get("category") == "VD"]
    vs_preds = [p for p in predictions if p.get("category") == "VS"]
    vd_acc = sum(p["correct"] for p in vd_preds) / len(vd_preds) if vd_preds else 0.0
    vs_acc = sum(p["correct"] for p in vs_preds) / len(vs_preds) if vs_preds else 0.0

    return {
        "aAcc": aacc,
        "vd_acc": vd_acc,
        "vs_acc": vs_acc,
        "total": total,
        "correct": correct,
    }
