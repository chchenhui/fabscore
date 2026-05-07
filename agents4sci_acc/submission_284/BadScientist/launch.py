#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import os
import logging

if __package__ is None or __package__ == "":
    # Allow running as `python launch.py`
    sys.path.append(str(Path(__file__).resolve().parent))
    from core import run_pipeline  # type: ignore
else:
    # If part of a package, prefer relative import
    from .core import run_pipeline


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--template-dir", required=True)
    p.add_argument("--seed-path", required=True)
    p.add_argument("--num-ideas", type=int, default=3)
    p.add_argument("--out", default="results")
    p.add_argument(
        "--enable-review",
        action="store_true",
        help="Enable LLM review step (disabled by default).",
    )
    args = p.parse_args()

    seed = json.loads(Path(args.seed_path).read_text(encoding="utf-8"))
    logging.info(
        f"Launching with template={args.template_dir}, seed={args.seed_path}, num_ideas={args.num_ideas}, out={args.out}"
    )
    out_dir = run_pipeline(
        args.template_dir,
        seed,
        args.out,
        num_ideas=args.num_ideas,
        enable_review=args.enable_review,
    )
    logging.info(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
