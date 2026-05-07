#!/usr/bin/env bash
set -euo pipefail

# One-shot LaTeX build (requires pdflatex/bibtex)
cd "$(dirname "$0")/../paper"

echo "[build] Running pdflatex..."
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null || true
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null

echo "[build] Done. Output: paper/main.pdf"

# Optional page-count check (requires pdfinfo)
if command -v pdfinfo >/dev/null 2>&1; then
  pages=$(pdfinfo main.pdf | awk '/Pages:/ {print $2}')
  echo "[build] PDF has $pages pages total (statements after references do not count toward the 8-page limit)."
else
  echo "[build] Install 'pdfinfo' to auto-check page count."
fi
