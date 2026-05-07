"""BEIR dataset loader wrapper.

Downloads and loads corpus, queries, and qrels for the 4 target BEIR datasets
(SciFact, TREC-COVID, FiQA-2018, ArguAna) using beir.util and GenericDataLoader.
"""

import os
import pathlib

from beir import util
from beir.datasets.data_loader import GenericDataLoader

DATASETS = ["scifact", "trec-covid", "fiqa", "arguana"]

DEFAULT_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "beir"


def download_dataset(dataset_name: str, data_dir: str | pathlib.Path | None = None) -> str:
    data_dir = pathlib.Path(data_dir or DEFAULT_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / dataset_name
    if dataset_path.exists():
        return str(dataset_path)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    out_dir = util.download_and_unzip(url, str(data_dir))
    return out_dir


def load_dataset(dataset_name: str, data_dir: str | pathlib.Path | None = None, split: str = "test"):
    dataset_path = download_dataset(dataset_name, data_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(split=split)
    return corpus, queries, qrels


def load_all_datasets(data_dir: str | pathlib.Path | None = None, split: str = "test"):
    results = {}
    for name in DATASETS:
        corpus, queries, qrels = load_dataset(name, data_dir, split)
        results[name] = {"corpus": corpus, "queries": queries, "qrels": qrels}
    return results
