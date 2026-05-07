
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_auroc(labels: list[int], scores: list[float]) -> float:
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC).

    Args:
        labels (list[int]): The true labels (0 or 1).
        scores (list[float]): The prediction scores from a detector.

    Returns:
        float: The AUROC score.
    """
    if len(np.unique(labels)) < 2:
        logging.warning("Only one class present in labels. AUROC is not defined.")
        return 0.0
    return roc_auc_score(labels, scores)

def calculate_fnr_at_fpr(labels: list[int], scores: list[float], target_fpr: float = 0.05) -> tuple[float, float, float, float]:
    """
    Calculates the False Negative Rate (FNR) at a specific False Positive Rate (FPR).
    
    Uses conservative operating point selection: picks the rightmost ROC point with FPR ≤ target.
    This ensures we stay within the FPR constraint for safety-critical applications.

    Args:
        labels (list[int]): The true labels (0 or 1).
        scores (list[float]): The prediction scores from a detector.
        target_fpr (float): The target maximum FPR at which to calculate FNR.

    Returns:
        tuple[float, float, float, float]: (FNR, threshold, FPR_used, TPR_used) for full auditability.
    """
    # Use drop_intermediate=False to preserve all useful operating points
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    
    try:
        # Conservative approach: find rightmost (highest TPR) point with FPR ≤ target
        valid_indices = np.where(fpr <= target_fpr)[0]
        
        if len(valid_indices) > 0:
            # Pick the rightmost valid point (highest TPR among valid FPRs)
            idx = valid_indices[-1]
            logging.info(f"Using conservative operating point: FPR={fpr[idx]:.6f} ≤ target={target_fpr:.6f}, TPR={tpr[idx]:.6f}")
        else:
            # No points satisfy FPR ≤ target
            # Fallback: pick the point with smallest FPR (most conservative available)
            idx = np.argmin(fpr)
            logging.warning(f"Target FPR {target_fpr:.6f} not achievable. Using most conservative point: FPR={fpr[idx]:.6f}, TPR={tpr[idx]:.6f}")
        
        threshold = thresholds[idx]
        fnr = 1 - tpr[idx]
        fpr_used = fpr[idx]
        tpr_used = tpr[idx]
        
        # Handle infinite threshold (only for logging/interpretability)
        if np.isinf(threshold):
            logging.info(f"Selected operating point has infinite threshold (perfect separation)")
        
        logging.info(f"Final metrics: FNR={fnr:.6f}, threshold={threshold}, FPR_used={fpr_used:.6f}, TPR_used={tpr_used:.6f}")
        
        return fnr, threshold, fpr_used, tpr_used
        
    except (IndexError, ValueError) as e:
        logging.warning(f"Could not determine FNR at {target_fpr * 100}% FPR: {e}")
        return 1.0, -1.0, 1.0, 0.0 # Return worst-case values

if __name__ == '__main__':
    # Example Usage
    # Mock data: 5 positive and 5 negative samples
    true_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # Mock scores where the detector correctly ranks most positives higher than negatives
    detector_scores = [0.1, 0.2, 0.3, 0.25, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9]

    auroc = calculate_auroc(true_labels, detector_scores)
    fnr_at_5_fpr, threshold, fpr_used, tpr_used = calculate_fnr_at_fpr(true_labels, detector_scores, target_fpr=0.05)

    print(f"--- Evaluation Metrics Example ---")
    print(f"True Labels: {true_labels}")
    print(f"Detector Scores: {detector_scores}")
    print(f"AUROC: {auroc:.4f}")
    print(f"FNR @ 5% FPR: {fnr_at_5_fpr:.4f} (threshold={threshold:.4f}, actual_FPR={fpr_used:.4f}, actual_TPR={tpr_used:.4f})")

    # Example with perfect separation
    perfect_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    auroc_perfect = calculate_auroc(true_labels, perfect_scores)
    print(f"\nAUROC (Perfect Separation): {auroc_perfect:.4f}")
