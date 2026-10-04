#!/bin/sh
# Rebuild the report from scratch.
#
#   ../.venv/bin/python ../src/report_figs.py   # regenerate figures first
#   ./build.sh
#
# Three pdflatex passes resolve the cross-references and float placement.
#
# NOTE on siunitx: this document deliberately does NOT use it. On the TeX Live
# build here, siunitx's expl3 machinery stalls pdflatex indefinitely -- no
# error, no output, the log simply stops mid-package. Units are formatted with
# a two-line \qty macro instead, which is all this document needed.
set -e
export PATH="/usr/local/bin:/Library/TeX/texbin:$PATH"
rm -f report.aux report.log report.out report.toc
pdflatex -interaction=nonstopmode -draftmode report.tex < /dev/null > /dev/null
pdflatex -interaction=nonstopmode -draftmode report.tex < /dev/null > /dev/null
pdflatex -interaction=nonstopmode report.tex < /dev/null > build.log
if [ -f report.pdf ]; then
  echo "built report.pdf ($(wc -c < report.pdf) bytes)"
  grep -c '^!' build.log > /dev/null 2>&1 && n=$(grep -c '^!' build.log) || n=0
  echo "errors: $n"
else
  echo "BUILD FAILED"; grep -A5 '^!' build.log | head -20; exit 1
fi
