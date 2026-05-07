# Online LaTeX Compilation Instructions

Since LaTeX is not installed locally, you can compile the paper online using these services:

## Recommended: Overleaf (Free)

1. Go to https://www.overleaf.com/
2. Create a free account
3. Click "New Project" → "Upload Project"
4. Upload these files from the `paper/` directory:
   - `main.tex`
   - `refs.bib`
   - `math_formulation.tex`
   - `agents4science_2025.sty`
   - All files from `figures/` folder
   - All files from `statements/` folder

## Alternative: ShareLaTeX or Other Online Editors

- **ShareLaTeX**: https://www.sharelatex.com/
- **LaTeX Base**: https://latexbase.com/
- **CoCalc**: https://cocalc.com/

## File Upload Order

1. **Main files**: `main.tex`, `refs.bib`, `agents4science_2025.sty`
2. **Figures**: Upload all PDF files from `figures/` directory
3. **Statements**: Upload all `.tex` files from `statements/` directory
4. **Math**: Upload `math_formulation.tex`

## Compilation Steps

1. Set main document to `main.tex`
2. Use pdfLaTeX compiler
3. Compile sequence:
   - pdflatex main.tex
   - bibtex main
   - pdflatex main.tex
   - pdflatex main.tex

## Expected Output

- **File**: main.pdf
- **Length**: ~7 pages
- **Content**: Complete paper with figures and references
- **Quality**: Publication-ready PDF

## Troubleshooting

If compilation fails:
1. Check all files are uploaded correctly
2. Ensure figures are in the right directory structure
3. Verify `agents4science_2025.sty` is in the main directory
4. Use pdfLaTeX (not XeLaTeX or LuaLaTeX)

The paper is designed to compile cleanly with standard LaTeX packages.