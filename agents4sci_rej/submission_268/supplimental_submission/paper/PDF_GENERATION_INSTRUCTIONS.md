# PDF Generation Instructions

## Quick Start

To generate the anonymized PDF paper, run:

```bash
cd paper/
./compile_paper.sh
```

## Manual Compilation

If you prefer manual compilation:

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## LaTeX Installation

### macOS
```bash
brew install --cask mactex
```

### Ubuntu/Debian
```bash
sudo apt-get install texlive-full
```

### Windows
Download and install MiKTeX from: https://miktex.org/

## Required Files

The paper directory contains all necessary files:
- `main.tex` - Main paper source
- `refs.bib` - Bibliography with real citations
- `math_formulation.tex` - Mathematical formulations
- `agents4science_2025.sty` - Conference style file
- `figures/` - All paper figures (5 PDF files)
- `statements/` - Required ethical statements

## Expected Output

The compilation will generate:
- `main.pdf` - Complete 7-page anonymized paper
- Includes all figures, tables, and references
- Ready for conference submission

## Troubleshooting

### Common Issues

1. **Missing packages**: Install texlive-full or complete MiKTeX
2. **Figure not found**: Ensure all PDF figures are in `figures/` directory
3. **Style file error**: Verify `agents4science_2025.sty` is in paper directory
4. **Bibliography issues**: Run bibtex between pdflatex runs

### Verification

After compilation, verify the PDF contains:
- ✅ Title page with anonymized author
- ✅ Abstract (192 words)
- ✅ All sections (Introduction through Conclusion)
- ✅ 2 figures (ablation study and trade-off plot)
- ✅ 1 results table
- ✅ Bibliography with 8 real references
- ✅ AI contribution disclosure
- ✅ Broader impact statement
- ✅ Total length: ~7 pages

## File Structure After Compilation

```
paper/
├── main.tex              # Source file
├── main.pdf              # Generated paper ✓
├── main.aux              # LaTeX auxiliary (auto-generated)
├── main.bbl              # Bibliography (auto-generated)
├── main.blg              # Bibliography log (auto-generated)
├── refs.bib              # Bibliography source
├── agents4science_2025.sty  # Style file
├── figures/              # Paper figures
│   ├── ablation_study.pdf
│   ├── fairness_accuracy_tradeoff.pdf
│   ├── confusion_matrices.pdf
│   ├── dataset_visualization.pdf
│   └── fairness_metrics_comparison.pdf
└── statements/           # Ethical statements
    ├── ai_contribution.tex
    ├── broader_impact.tex
    └── reproducibility.tex
```

## Quality Checklist

Before submission, verify:
- [ ] PDF compiles without errors
- [ ] All figures display correctly
- [ ] Bibliography formats properly
- [ ] Page count ≤ 8 pages
- [ ] No author identifying information
- [ ] Mathematical notation renders correctly
- [ ] Hyperlinks work properly