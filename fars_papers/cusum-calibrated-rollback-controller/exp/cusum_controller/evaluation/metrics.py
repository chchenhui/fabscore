# Metric computation and result aggregation.
# Implements: y_pre, peak excess loss, excess AUC, rollback counts, detection delay.
# All metrics computed from the probe loss trajectory y_t for t=0..250.

import numpy as np


def compute_y_pre(probe_losses, pre_start=110, pre_end=119):
    return np.mean(probe_losses[pre_start : pre_end + 1])


def compute_peak_excess_loss(probe_losses, y_pre, perturb_start=120):
    return float(np.max(probe_losses[perturb_start:] - y_pre))


def compute_excess_auc(probe_losses, y_pre, perturb_start=120, end=251):
    excess = np.maximum(0.0, probe_losses[perturb_start:end] - y_pre)
    return float(np.sum(excess))


def compute_nominal_rollback_fraction(decisions):
    rollbacks = np.sum(~decisions[1:])
    total = len(decisions) - 1
    return float(rollbacks / total) if total > 0 else 0.0


def compute_false_rollbacks_outside_window(decisions, window_start=120, window_len=10):
    count = 0
    for t in range(1, len(decisions)):
        step = t - 1
        if not decisions[t] and not (window_start <= step < window_start + window_len):
            count += 1
    return count


def compute_detection_delay(decisions, window_start=120):
    for t in range(1, len(decisions)):
        step = t - 1
        if not decisions[t]:
            return step - window_start
    return None


def compute_all_metrics(result_dict, window_start=120, window_len=10):
    probe_losses = result_dict["probe_losses"]
    decisions = result_dict["decisions"]
    perturb_type = result_dict["perturb_type"]

    y_pre = compute_y_pre(probe_losses)

    metrics = {"y_pre": float(y_pre)}

    if perturb_type != "nominal":
        metrics["peak_excess_loss"] = compute_peak_excess_loss(probe_losses, y_pre)
        metrics["excess_auc"] = compute_excess_auc(probe_losses, y_pre)
        metrics["false_rollbacks_outside_window"] = compute_false_rollbacks_outside_window(
            decisions, window_start, window_len
        )
        metrics["detection_delay"] = compute_detection_delay(decisions, window_start)
    else:
        metrics["nominal_rollback_fraction"] = compute_nominal_rollback_fraction(decisions)

    return metrics


def aggregate_across_seeds(metric_dict_list):
    if not metric_dict_list:
        return {}

    all_keys = set()
    for m in metric_dict_list:
        all_keys.update(m.keys())

    aggregated = {}
    for key in all_keys:
        values = [m[key] for m in metric_dict_list if key in m and m[key] is not None]
        if len(values) == 0:
            aggregated[key] = {"mean": None, "std": None, "count": 0}
        else:
            arr = np.array(values, dtype=np.float64)
            aggregated[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "count": len(arr),
            }

    return aggregated
