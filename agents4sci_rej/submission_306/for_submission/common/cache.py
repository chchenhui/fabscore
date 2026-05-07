import os
import json
import time
from typing import Optional, Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Optional[Any]:
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception:
        return None
    return None


def save_json(path: str, obj: Any) -> None:
    try:
        ensure_dir(os.path.dirname(path))
        with open(path, 'w') as f:
            json.dump(obj, f)
    except Exception:
        pass


def cache_path(root: str, name: str) -> str:
    ensure_dir(root)
    safe = ''.join([c if c.isalnum() or c in ('_', '-', '.') else '_' for c in name])
    return os.path.join(root, safe)

