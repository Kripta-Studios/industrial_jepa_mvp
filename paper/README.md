# Academic Paper Compilation Guide

This folder contains the academic LaTeX paper documenting the findings of the `industrial_jepa_mvp` project.

## Files
* `main.tex`: Academic article source (English).
* `references.bib`: BibTeX citations.

## How to Compile

To compile the paper to PDF, ensure you have a standard LaTeX distribution installed (such as TeX Live, MiKTeX, or MacTeX).

### Option 1: Standard pdflatex + bibtex
Run the following commands in sequence to compile the document and generate the correct citation indexes:
```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Option 2: Automatic latexmk
If `latexmk` is installed, simply run:
```powershell
latexmk -pdf main.tex
```

### Missing LaTeX Tools Fallback
If no LaTeX tools are installed on your current system, you can upload `main.tex` and `references.bib` to an online LaTeX compiler such as Overleaf (https://www.overleaf.com) to generate the PDF automatically.
