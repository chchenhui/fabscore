"""Preprocess LogHub 2k datasets (BGL, Linux, HDFS) into JSONL format.
Reads *_2k.log_structured.csv for each dataset, extracts (Content, EventTemplate,
EventId) triples, and writes to data/processed/{dataset}.jsonl.
"""

import argparse
import csv
import json
import os
from pathlib import Path


DATASETS = ["BGL", "Linux", "HDFS"]

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw" / "loghub"
OUT_DIR = ROOT / "processed"


def preprocess_dataset(dataset: str) -> int:
    csv_path = RAW_DIR / dataset / f"{dataset}_2k.log_structured.csv"
    out_path = OUT_DIR / f"{dataset}.jsonl"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "log_id": int(row["LineId"]),
                "raw_log": row["Content"],
                "template": row["EventTemplate"],
                "event_id": row["EventId"],
            })

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Preprocess LogHub datasets")
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        help="Datasets to preprocess (default: BGL Linux HDFS)",
    )
    args = parser.parse_args()

    for ds in args.datasets:
        n = preprocess_dataset(ds)
        print(f"{ds}: {n} records -> {OUT_DIR / f'{ds}.jsonl'}")


if __name__ == "__main__":
    main()
