#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v pdflatex >/dev/null || { echo "pdfLaTeX is required" >&2; exit 1; }
if command -v bibtex >/dev/null; then BIBTEX=bibtex
elif command -v bibtex.original >/dev/null; then BIBTEX=bibtex.original
else echo "BibTeX is required" >&2; exit 1; fi
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
"$BIBTEX" manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
