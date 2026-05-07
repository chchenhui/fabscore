#!/usr/bin/env python
"""
Review exactly one PDF using the repository's review logic (as in core.py).
Outputs are saved under external_reviews/<pdf_basename>/.

Usage:
  python tools/review_one_pdf.py /path/to/paper.pdf [--out external_reviews]
"""

import argparse
import json
import os
import sys
from datetime import datetime


def _import_review_details_sym():
    # Try to import a diagnostics object that includes individual model reviews
    try:
        from ..perform_review import LAST_REVIEW_DETAILS  # type: ignore

        return LAST_REVIEW_DETAILS
    except Exception:
        pass
    try:
        from perform_review import LAST_REVIEW_DETAILS  # type: ignore

        return LAST_REVIEW_DETAILS
    except Exception:
        return None


def _import_review_helpers():
    # Try package-style import first (when installed as a module)
    try:
        from ..perform_review import perform_review as run_review, load_paper  # type: ignore

        return run_review, load_paper
    except Exception:
        pass

    # Try repo-root import when run as a script
    try:
        from perform_review import perform_review as run_review, load_paper  # type: ignore

        return run_review, load_paper
    except Exception:
        pass

    # Last resort: add repo root to sys.path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    from perform_review import perform_review as run_review, load_paper  # type: ignore

    return run_review, load_paper


def review_one_pdf(
    pdf_path: str,
    out_root: str = "external_reviews",
    num_fs_examples: int = 3,
    temperature: float = 1.0,
) -> str:
    run_review, load_paper = _import_review_helpers()

    if not isinstance(pdf_path, str) or not pdf_path.lower().endswith(".pdf"):
        raise ValueError("Provide exactly one .pdf file path")

    abs_pdf = os.path.abspath(pdf_path)
    if not os.path.exists(abs_pdf):
        # Small convenience: if a relative filename was provided, try ./pdfs/<name>.pdf
        if not os.path.isabs(pdf_path):
            cand = os.path.abspath(os.path.join("pdfs", pdf_path))
            if os.path.exists(cand):
                abs_pdf = cand
        if not os.path.exists(abs_pdf):
            raise FileNotFoundError(abs_pdf)

    base = os.path.splitext(os.path.basename(abs_pdf))[0]
    out_dir = os.path.join(out_root, base)
    os.makedirs(out_dir, exist_ok=True)

    # Load text from the PDF (same helper core uses)
    text = load_paper(abs_pdf)

    # Mirror core perform_review behavior; few-shot and temperature forwarded
    review = run_review(
        text,
        model=None,
        client=None,
        num_reflections=1,
        num_fs_examples=num_fs_examples,
        num_reviews_ensemble=1,
        temperature=temperature,
    )

    # Prepare output JSON: always include aggregated review under 'review'.
    # If details available (per-model outputs), include them for debugging/analysis.
    details = _import_review_details_sym()
    out_obj = {"review": review}
    try:
        if details and isinstance(details, dict):
            models = details.get("models")
            indiv = details.get("individual_reviews")
            if indiv and len(indiv) > 0:
                out_obj["models"] = models
                out_obj["individual_reviews"] = indiv
                # Only add aggregated if it's different from the main review
                aggregated = details.get("aggregated")
                if aggregated and aggregated != review:
                    out_obj["aggregated"] = aggregated
    except Exception:
        pass

    # Match core's output filename
    out_json_path = os.path.join(out_dir, "review.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    # Also write a minimal text summary for quick inspection
    try:
        lines = [
            f"PDF: {abs_pdf}",
            f"Reviewed: {datetime.utcnow().isoformat()}Z",
            f"Decision: {review.get('Decision', 'N/A')}",
            f"Overall: {review.get('Overall', 'N/A')} | Confidence: {review.get('Confidence', 'N/A')}",
            "",
            "Summary:",
            str(review.get("Summary", "")),
        ]
        with open(os.path.join(out_dir, "review.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass

    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Review exactly one PDF and save results")
    ap.add_argument("pdf", help="Path to a single PDF file to review")
    ap.add_argument("--out", default="external_reviews", help="Output root directory")
    ap.add_argument(
        "--fewshot",
        type=int,
        default=3,
        help="Number of few-shot examples to include (0-3)",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default matches core).",
    )
    args = ap.parse_args(argv)

    out_dir = review_one_pdf(
        args.pdf,
        out_root=args.out,
        num_fs_examples=args.fewshot,
        temperature=args.temperature,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
