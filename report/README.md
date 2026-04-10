# LaTeX Report Template — File Structure & Compilation Guide

## Directory Layout

```
report_template/
├── main.tex                        ← Compile THIS file
├── references.bib                  ← BibTeX bibliography
└── sections/
    ├── 00_titlepage.tex            ← Title page
    ├── 01_executive_summary.tex    ← Executive Summary
    ├── 02_introduction.tex         ← Introduction
    ├── 03_procedure.tex            ← Procedure (with subsections)
    ├── 04_results.tex              ← Results (matching subsections)
    ├── 05_conclusions.tex          ← Conclusions
    ├── 06_appendices.tex           ← Appendices
    └── 07_addendum.tex             ← Addendum (contribution letters)
```

## How to Compile

### Using the command line (pdflatex)
Run these commands in the `report_template/` directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
> Three pdflatex passes are needed to resolve cross-references and the TOC.

### Using latexmk (recommended — handles passes automatically)
```bash
latexmk -pdf main.tex
```

### Using an IDE
Open `main.tex` in **Overleaf**, **TeXstudio**, **VSCode + LaTeX Workshop**,
or **TeXmaker** and compile with the default PDF recipe.

## Customisation Tips

| Task | Where to change |
|------|----------------|
| Authors, title, date | `sections/00_titlepage.tex` |
| Margins | `\geometry{margin=...}` in `main.tex` |
| Font size | `\documentclass[12pt,...]` in `main.tex` |
| Citation style | `\bibliographystyle{...}` in `main.tex` |
| Add a subsection to Procedure/Results | Add `\section{...}` in the relevant file |
| Add a group member letter | Duplicate a letter block in `07_addendum.tex` |
| Remove List of Figures/Tables | Delete `\listoffigures` / `\listoftables` in `main.tex` |

## Placeholder Text
All `\lipsum[...]` commands produce dummy Latin text.
Replace them with your actual content and remove the
`\usepackage{lipsum}` line from `main.tex` when done.
