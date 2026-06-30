# Industrial World Model Technical Report

Compile with:

```powershell
cd paper_iwm
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Current output:

* `main.tex`: expanded technical report with implementation details, dataset manifest summary, DINOv2/PatchCore/PaDiM visual results, sensor evidence, LeJEPA diagnostics and LeWorldModel smoke results.
* `main.pdf`: compiled PDF.
* `references.bib`: conservative BibTeX references without invented DOI fields.

This is a technical report for MVP validation and commercial pilot support, not a final academic publication.
