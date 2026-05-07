"""
Utilities to make LaTeX figure includes robust by ensuring all referenced
assets exist and render safely. Default policy:
    - If an image is missing or invalid, remove the \includegraphics entirely.
    - If an image is "mostly dark" (near-black), also remove the \includegraphics.
    - Otherwise, keep it unchanged.

This avoids black boxes or compilation failures from corrupt PNGs.

Typical usage:
        from tools.figure_safety import scan_and_fix_graphics
        report = scan_and_fix_graphics('/path/to/latex/template.tex')
"""

from __future__ import annotations

import os
import os.path as osp
import re
from typing import Dict, List, Tuple


# Minimal valid 1x1 PNG bytes
_PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc````\x00\x00\x00\x04\x00\x01"
    b"\x0b\xe7\x02\x9e\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _is_valid_image(path: str) -> bool:
    """Best-effort validation for PNG/JPEG/PDF.
    - PNG: magic 89 50 4E 47 0D 0A 1A 0A
    - JPEG: magic FF D8 at start and FF D9 at end
    - PDF: starts with %PDF-
    """
    if not osp.exists(path) or osp.getsize(path) < 4:
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            if head.startswith(b"%PDF-"):
                return True
            if head == b"\x89PNG\r\n\x1a\n":
                return True
            if head[:2] == b"\xff\xd8":
                # check jpeg tail
                f.seek(-2, os.SEEK_END)
                tail = f.read(2)
                return tail == b"\xff\xd9"
    except Exception:
        return False
    return False


def _write_placeholder_png(path: str) -> None:
    """Deprecated: kept for backward compatibility if needed elsewhere."""
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(_PNG_1x1)


def _is_mostly_dark(path: str, threshold: float = 0.06) -> bool:
    """Return True if average luminance is below threshold in [0,1].
    Uses matplotlib + numpy to avoid new heavy deps.
    """
    try:
        import matplotlib.image as mpimg  # type: ignore
        import numpy as np  # type: ignore

        arr = mpimg.imread(path)
        if arr is None:
            return False
        # Normalize to [0,1]
        if arr.dtype.kind in ("u", "i") and arr.max() > 1.0:
            arr = arr.astype(np.float32) / 255.0
        if arr.ndim == 2:
            lum = arr
        else:
            rgb = arr[..., :3]
            lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        mean_lum = float(lum.mean())
        return mean_lum < threshold
    except Exception:
        return False


def _collect_graphics_paths(tex_src: str) -> List[Tuple[str, Tuple[int, int]]]:
    """Return list of (path, (start_index, end_index)) for includegraphics braces.
    The start/end indices refer to the path substring within the full source.
    """
    results: List[Tuple[str, Tuple[int, int]]] = []
    # This regex captures path inside braces of \includegraphics[<opts>]{path}
    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    for m in pattern.finditer(tex_src):
        path = m.group(1)
        # compute index of the captured group within the full string
        start = m.start(1)
        end = m.end(1)
        results.append((path, (start, end)))
    return results


def scan_and_fix_graphics(tex_path: str) -> Dict[str, List[str]]:
    """Scan a LaTeX file for \includegraphics and sanitize includes.

    Default policy:
        - If referenced asset is missing or invalid: remove the \includegraphics.
        - If asset exists and is valid but mostly dark (avg luminance < 0.06): remove the include.
        - Otherwise: keep as-is.

    Returns a report dict with keys: 'hidden', 'unchanged'.
    (Keys 'fixed'/'generated' left for backward compatibility but empty.)
    """
    tex_path = osp.abspath(tex_path)
    latex_dir = osp.dirname(tex_path)
    with open(tex_path, "r", encoding="utf-8", errors="replace") as f:
        src = f.read()

    includes = _collect_graphics_paths(src)
    if not includes:
        return {"fixed": [], "generated": [], "unchanged": [], "hidden": []}

    fixed: List[str] = []
    generated: List[str] = []
    unchanged: List[str] = []
    hidden: List[str] = []
    # map original rel path string -> replacement path or None (None = remove entire include)
    replacements: Dict[str, str] = {}

    for rel_path, _ in includes:
        # Normalize and resolve
        rel_path_norm = rel_path.replace("\\", "/")
        abs_path = osp.normpath(osp.join(latex_dir, rel_path_norm))

        if not _is_valid_image(abs_path):
            # hide missing/invalid
            hidden.append(rel_path)
            # mark for removal (handled in callback)
            replacements[rel_path] = None  # type: ignore
            continue

        # valid image; check darkness
        try:
            if _is_mostly_dark(abs_path):
                hidden.append(rel_path)
                replacements[rel_path] = None  # type: ignore
                continue
        except Exception:
            # if brightness check fails, keep as-is
            pass

        unchanged.append(rel_path)

    if replacements:
        # Rebuild file using regex with callback to safely replace only the path group
        regex = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})")

        def _cb(m: re.Match) -> str:
            path = m.group(2)
            if path in replacements:
                new_path = replacements[path]
                if new_path is None:
                    # remove the entire includegraphics command
                    return ""
                # (kept for backward compatibility; currently unused)
                return m.group(1) + new_path + m.group(3)
            return m.group(0)

        new_src = regex.sub(_cb, src)
        if new_src != src:
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(new_src)

    return {
        "fixed": fixed,
        "generated": generated,
        "unchanged": unchanged,
        "hidden": hidden,
    }
