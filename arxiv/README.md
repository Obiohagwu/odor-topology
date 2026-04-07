# arXiv Bundle

This directory contains an arXiv-oriented source draft for the odor-topology analysis.

## Contents

- `main.tex`: manuscript source
- `figures/`: generated figure assets used by the manuscript

The manuscript currently includes:

- filled-in author metadata derived from the local git config
- an inline bibliography so the arXiv bundle does not depend on an external `.bib` or `.bbl`
- direct figure references that resolve within this directory

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

## Upload packaging

To build a clean upload zip from the repo root without macOS metadata files:

```bash
cd /Users/oboh/bio-research/odor-topology
zip -r arxiv.zip arxiv -x "*/.DS_Store"
```

## Final submission checks

- verify the author name and email in `main.tex` match your preferred public metadata
- compile the TeX source once a TeX toolchain is available
- inspect the generated PDF for line breaks, figure placement, and hyperlink formatting
