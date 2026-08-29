#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

exec latexmk \
  -pvc \
  -view=none \
  -pdf \
  -outdir=paper \
  -interaction=nonstopmode \
  -halt-on-error \
  paper/dataset_paper_v4.tex
