# Embedding wrapper supporting BAAI/bge-m3 (via FlagEmbedding) and
# Qwen/Qwen3-Embedding-4B (via sentence-transformers) backends.
# Provides record-level and query-level encoding with disk caching.

import hashlib
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        if self.model_name == "BAAI/bge-m3":
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(self.model_name, use_fp16=True)
        elif "Qwen" in self.model_name:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _cache_key(self, prefix: str, texts: List[str]) -> str:
        h = hashlib.md5(json.dumps(texts, sort_keys=True).encode()).hexdigest()[:12]
        safe_name = self.model_name.replace("/", "_")
        return f"{prefix}_{safe_name}_{h}"

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"embeddings_{key}.npy"

    def encode(self, records: List[Dict], cache_prefix: str = "records",
               force: bool = False) -> np.ndarray:
        texts = [r["req"] + "\n" + r["resp"] for r in records]
        cache_key = self._cache_key(cache_prefix, texts)
        cache_file = self._cache_path(cache_key)

        if not force and cache_file.exists():
            emb = np.load(cache_file)
            if emb.shape[0] == len(texts):
                return emb

        self._load_model()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if self.model_name == "BAAI/bge-m3":
            out = self._model.encode(texts, batch_size=32, max_length=8192)
            vecs = out["dense_vecs"]
        else:
            vecs = self._model.encode(texts, batch_size=16, show_progress_bar=True)

        vecs = np.asarray(vecs, dtype=np.float32)
        np.save(cache_file, vecs)
        return vecs

    def encode_queries(self, queries: List[str], cache_prefix: str = "queries",
                       force: bool = False) -> np.ndarray:
        cache_key = self._cache_key(cache_prefix, queries)
        cache_file = self._cache_path(cache_key)

        if not force and cache_file.exists():
            emb = np.load(cache_file)
            if emb.shape[0] == len(queries):
                return emb

        self._load_model()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if self.model_name == "BAAI/bge-m3":
            out = self._model.encode(queries, batch_size=32, max_length=8192)
            vecs = out["dense_vecs"]
        elif "Qwen" in self.model_name:
            vecs = self._model.encode(queries, batch_size=16,
                                      prompt_name="query", show_progress_bar=True)
        else:
            vecs = self._model.encode(queries, batch_size=16, show_progress_bar=True)

        vecs = np.asarray(vecs, dtype=np.float32)
        np.save(cache_file, vecs)
        return vecs
