#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_TEX="$REPO_ROOT/paper/paper.tex"
OUTPUT_PDF="$REPO_ROOT/output/pdf/dancing-stick-figures-arxiv-v1-candidate.pdf"
OUTPUT_TAR="$REPO_ROOT/output/dancing-stick-figures-arxiv-v1-source.tar.gz"

BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/dsf-arxiv.XXXXXX")"
trap 'rm -rf "$BUILD_ROOT"' EXIT
STAGE="$BUILD_ROOT/submission"
mkdir -p "$STAGE/figs" "$REPO_ROOT/output/pdf"

# arXiv receives paper.tex at the archive root, so remove the repository-only
# paper/ prefix from figure paths.
sed 's#paper/figs/#figs/#g' "$SOURCE_TEX" > "$STAGE/paper.tex"

for figure in dataset_anatomy failure_modes reference_generation_pairs; do
  cp "$REPO_ROOT/paper/figs/${figure}.pdf" "$STAGE/figs/${figure}.pdf"
done

(
  cd "$STAGE"
  latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex >/dev/null
)

if rg -n 'LaTeX Warning: (Citation|Reference).*undefined|There were undefined references|multiply defined' \
  "$STAGE/paper.log"; then
  echo "arXiv compile contains unresolved references" >&2
  exit 1
fi

cp "$STAGE/paper.pdf" "$OUTPUT_PDF"
(
  cd "$STAGE"
  shasum -a 256 paper.tex figs/*.pdf > SHA256SUMS
  tar -czf "$OUTPUT_TAR" paper.tex figs SHA256SUMS
)

echo "$OUTPUT_PDF"
echo "$OUTPUT_TAR"

