# arXiv Bundle

This directory contains an arXiv-oriented source draft for the odor-topology analysis.

## Contents

- `main.tex`: manuscript source
- `figures/`: generated figure assets used by the manuscript

## Regenerate figures

Run:

```bash
/Users/oboh/bio-research/odor-topology/.venv/bin/python \
  /Users/oboh/bio-research/odor-topology/scripts/14_make_arxiv_figures.py
```

## Build note

The local packaging pass that created this bundle did not have `pdflatex` or `latexmk` installed, so the TeX source was prepared but not compiled on this machine.

Once a TeX toolchain is available, a standard build command is:

```bash
cd /Users/oboh/bio-research/odor-topology/arxiv
pdflatex main.tex
pdflatex main.tex
```

## Final submission edits

- replace the placeholder author line in `main.tex`
- add external literature citations if desired before upload
