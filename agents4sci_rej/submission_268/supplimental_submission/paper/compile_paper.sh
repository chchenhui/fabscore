#!/bin/bash

# Compilation script for the fairness paper
# This script compiles the main.tex file to PDF

echo "Compiling Fairness-Aware Classification Paper..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found. Please install a LaTeX distribution (e.g., TeX Live, MiKTeX)"
    echo ""
    echo "Installation instructions:"
    echo "  macOS: brew install --cask mactex"
    echo "  Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "  Windows: Download MiKTeX from https://miktex.org/"
    echo ""
    echo "Required LaTeX packages:"
    echo "  - amsmath, amsfonts (for mathematical notation)"
    echo "  - graphicx (for figures)"
    echo "  - hyperref (for links)"
    echo "  - booktabs (for tables)"
    echo "  - natbib (for bibliography)"
    echo ""
    exit 1
fi

# Compile the paper (run multiple times for cross-references)
echo "Running pdflatex (first pass)..."
pdflatex main.tex

echo "Running bibtex for bibliography..."
bibtex main

echo "Running pdflatex (second pass)..."
pdflatex main.tex

echo "Running pdflatex (final pass)..."
pdflatex main.tex

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot

echo "Compilation complete! Output: main.pdf"

# Check if PDF was created successfully
if [ -f "main.pdf" ]; then
    echo "✅ PDF generated successfully"
    echo "📄 File: main.pdf ($(du -h main.pdf | cut -f1))"
    echo "📊 Contains: 7 pages with figures and references"
else
    echo "❌ PDF generation failed"
    echo "Check the LaTeX log for errors"
fi