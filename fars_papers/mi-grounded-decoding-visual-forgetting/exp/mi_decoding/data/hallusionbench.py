# HallusionBench loader. Loads HallusionBench.json (1129 items).
# Items with visual_input != "0" have images; others are text-only (image=None).
# Image paths in JSON are like "./VD/figure/0_0.png", resolved under hallusion_bench/.
import json
import os
from PIL import Image


def load_hallusionbench(bench_dir):
    json_path = os.path.join(bench_dir, "HallusionBench.json")
    with open(json_path) as f:
        data = json.load(f)

    items = []
    for i, row in enumerate(data):
        image = None
        filename = row.get("filename")
        if filename:
            img_path = os.path.join(bench_dir, "hallusion_bench", filename.lstrip("./"))
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")

        item_id = f"{row['category']}_{row['subcategory']}_{row['set_id']}_{row['figure_id']}_{row['question_id']}"
        items.append({
            "id": item_id,
            "index": i,
            "image": image,
            "question": row["question"],
            "gt_answer": str(row["gt_answer"]),
            "category": row["category"],
            "subcategory": row["subcategory"],
            "set_id": row["set_id"],
            "figure_id": row["figure_id"],
            "question_id": row["question_id"],
            "visual_input": row["visual_input"],
            "filename": filename,
        })
    return items
