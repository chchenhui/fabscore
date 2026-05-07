"""Download and load the SIFT10K (siftsmall) dataset from the TexMex corpus.

Returns 10,000 base vectors of 128-d SIFT descriptors as float32 numpy array.
Caches to outputs/sift10k/vectors.npy after first download.
"""

import os
import struct
import tarfile
import tempfile
import urllib.request

import numpy as np

SIFT10K_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
SIFT10K_URL_HTTP = "http://corpus-texmex.irisa.fr/siftsmall.tar.gz"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "sift10k")
CACHE_PATH = os.path.join(OUTPUT_DIR, "vectors.npy")


def _read_fvecs(filepath: str) -> np.ndarray:
    vectors = []
    with open(filepath, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            d = struct.unpack("i", dim_bytes)[0]
            vec = np.frombuffer(f.read(d * 4), dtype=np.float32)
            vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def load_sift10k() -> np.ndarray:
    if os.path.exists(CACHE_PATH):
        return np.load(CACHE_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "siftsmall.tar.gz")

        for url in [SIFT10K_URL_HTTP, SIFT10K_URL]:
            try:
                print(f"Downloading SIFT10K from {url} ...")
                urllib.request.urlretrieve(url, tar_path)
                break
            except Exception as e:
                print(f"  Failed ({e}), trying next URL...")
        else:
            raise RuntimeError("Could not download SIFT10K from any mirror.")

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmpdir, filter="data")

        fvecs_path = os.path.join(tmpdir, "siftsmall", "siftsmall_base.fvecs")
        if not os.path.exists(fvecs_path):
            for root, dirs, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith("_base.fvecs"):
                        fvecs_path = os.path.join(root, f)
                        break

        vectors = _read_fvecs(fvecs_path)

    assert vectors.shape == (10000, 128), f"Unexpected shape: {vectors.shape}"
    np.save(CACHE_PATH, vectors)
    print(f"Saved SIFT10K vectors to {CACHE_PATH}")
    return vectors


if __name__ == "__main__":
    vecs = load_sift10k()
    print(f"SIFT10K: shape={vecs.shape}, dtype={vecs.dtype}")
