# Select representative RefCOCO val examples for attention visualization.
# Reconstructs per-sample results from sharded rank files (round-robin pattern)
# and selects 6-8 examples with spatial diversity where FastV fails but Ours succeeds.

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_JSONL = BASE_DIR / "data" / "refcoco" / "refcoco_val.jsonl"
RESULTS_DIR = BASE_DIR / "results"
OUT_DIR = BASE_DIR / "results" / "visualizations"

METHOD_CONFIGS = {
    "fastv": {"dir": "fastv_final", "split": "refcoco_val", "n_ranks": 8},
    "d2pruner": {"dir": "d2pruner", "split": "refcoco_val", "n_ranks": 8},
    "ours": {"dir": "opt_rawmis_km12", "split": "refcoco_val", "n_ranks": 4},
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def reconstruct_results(method_dir, split, n_ranks):
    results = {}
    for rank in range(n_ranks):
        rank_file = RESULTS_DIR / method_dir / f"{split}_rank{rank}.json"
        if not rank_file.exists():
            print(f"  WARNING: {rank_file} not found")
            continue
        with open(rank_file) as f:
            data = json.load(f)
        outputs = data.get("outputs", [])
        for i, item in enumerate(outputs):
            global_idx = rank + i * n_ranks
            results[global_idx] = item
    return results


def classify_position(bbox, width, height):
    cx = (bbox[0] + bbox[2]) / 2.0 / width
    cy = (bbox[1] + bbox[3]) / 2.0 / height
    if cy < 0.33:
        vpos = "top"
    elif cy > 0.66:
        vpos = "bottom"
    else:
        vpos = "center"
    if cx < 0.33:
        hpos = "left"
    elif cx > 0.66:
        hpos = "right"
    else:
        hpos = "center"
    return f"{vpos}-{hpos}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading refcoco_val JSONL...")
    jsonl_data = load_jsonl(DATA_JSONL)
    print(f"  Total samples: {len(jsonl_data)}")

    all_results = {}
    for method, cfg in METHOD_CONFIGS.items():
        print(f"Reconstructing {method} results ({cfg['n_ranks']} ranks)...")
        all_results[method] = reconstruct_results(cfg["dir"], cfg["split"], cfg["n_ranks"])
        print(f"  Reconstructed {len(all_results[method])} samples")

    common_indices = set(all_results["fastv"].keys()) & set(all_results["d2pruner"].keys()) & set(all_results["ours"].keys())
    common_indices = sorted([i for i in common_indices if i < len(jsonl_data)])
    print(f"Common indices across all methods: {len(common_indices)}")

    candidates = []
    for idx in common_indices:
        data = jsonl_data[idx]
        fastv_iou = all_results["fastv"][idx]["iou"]
        d2_iou = all_results["d2pruner"][idx]["iou"]
        ours_iou = all_results["ours"][idx]["iou"]

        bbox = data["bbox"]
        w, h = data["width"], data["height"]
        position = classify_position(bbox, w, h)

        fastv_fail = fastv_iou < 0.5
        ours_success = ours_iou >= 0.5
        d2_success = d2_iou >= 0.5

        if fastv_fail and ours_success:
            candidates.append({
                "jsonl_idx": idx,
                "image": data["image"],
                "sent": data["sent"],
                "bbox": bbox,
                "width": w,
                "height": h,
                "position": position,
                "fastv_iou": fastv_iou,
                "d2pruner_iou": d2_iou,
                "ours_iou": ours_iou,
                "d2_success": d2_success,
            })

    print(f"\nCandidates (FastV fails, Ours succeeds): {len(candidates)}")

    position_groups = {}
    for c in candidates:
        pos = c["position"]
        if pos not in position_groups:
            position_groups[pos] = []
        position_groups[pos].append(c)

    print("Position distribution of candidates:")
    for pos, items in sorted(position_groups.items()):
        print(f"  {pos}: {len(items)}")

    selected = []
    used_positions = set()
    target_positions = ["top-left", "top-center", "top-right",
                        "center-left", "center-center", "center-right",
                        "bottom-left", "bottom-center", "bottom-right"]

    for pos in target_positions:
        if pos in position_groups and pos not in used_positions:
            group = position_groups[pos]
            group_sorted = sorted(group, key=lambda x: (
                int(x["d2_success"]),
                x["ours_iou"],
                -x["fastv_iou"],
            ), reverse=True)
            best = group_sorted[0]
            selected.append(best)
            used_positions.add(pos)
            if len(selected) >= 8:
                break

    if len(selected) < 6:
        remaining = [c for c in candidates if c not in selected]
        remaining.sort(key=lambda x: (int(x["d2_success"]), x["ours_iou"] - x["fastv_iou"]), reverse=True)
        for c in remaining:
            selected.append(c)
            if len(selected) >= 6:
                break

    has_top = any("top" in s["position"] for s in selected)
    if not has_top:
        for c in candidates:
            if "top" in c["position"] and c not in selected:
                if len(selected) >= 8:
                    selected[-1] = c
                else:
                    selected.append(c)
                break

    print(f"\n=== Selected {len(selected)} examples ===")
    for i, s in enumerate(selected):
        print(f"  [{i}] idx={s['jsonl_idx']}, pos={s['position']}, "
              f"FastV={s['fastv_iou']:.3f}, D2={s['d2pruner_iou']:.3f}, Ours={s['ours_iou']:.3f}")
        print(f"       img={s['image']}, sent=\"{s['sent'][:60]}\"")

    out_path = OUT_DIR / "selected_examples.json"
    with open(out_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
