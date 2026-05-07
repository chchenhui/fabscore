# MMStar dataset loader. Loads Lin-Chen/MMStar (1500 val items) as list of dicts
# with keys: id, image (PIL), question (str), answer (str, one of A/B/C/D).
from datasets import load_dataset


def load_mmstar():
    ds = load_dataset("Lin-Chen/MMStar", split="val")
    items = []
    for i, row in enumerate(ds):
        items.append({
            "id": row.get("index", i),
            "image": row["image"],
            "question": row["question"],
            "answer": row["answer"],
        })
    return items
