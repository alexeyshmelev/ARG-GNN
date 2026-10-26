#!/usr/bin/env bash
# Compile only the manuscript and a freshly extracted source bundle.
set -euo pipefail
cd /workspace-SR003.nfs2/ARG-GNN
. scripts/project_env.sh
export PATH="$ARGGNN_ROOT/.runtime/tex/.TinyTeX/bin/x86_64-linux:$PATH"
export BIBINPUTS="$ARGGNN_ROOT/paper:"
export BSTINPUTS="$ARGGNN_ROOT/paper:"
figure_work="$ARGGNN_ROOT/paper/figure_redesign_20261026_v6"
cd "$ARGGNN_ROOT/paper"
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode -output-directory="$figure_work/build" current_full_panel_working.tex > "$figure_work/build/pass1.log" 2>&1
(
  cd "$figure_work/build"
  bibtex current_full_panel_working > bibtex.log 2>&1
)
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode -output-directory="$figure_work/build" current_full_panel_working.tex > "$figure_work/build/pass2.log" 2>&1
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode -output-directory="$figure_work/build" current_full_panel_working.tex > "$figure_work/build/pass3.log" 2>&1
cp "$figure_work/build/current_full_panel_working.pdf" current_full_panel_working.pdf
cd "$figure_work/overleaf_build"
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode main.tex > pass1.log 2>&1
bibtex main > bibtex.log 2>&1
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode main.tex > pass2.log 2>&1
pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode main.tex > pass3.log 2>&1
printf 'Canonical manuscript and extracted Overleaf source compiled.\n'
