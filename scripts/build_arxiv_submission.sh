#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_TEX="$REPO_ROOT/paper/dataset_paper_v6.tex"
OUTPUT_PDF="$REPO_ROOT/paper/dataset_paper_v6_arxiv.pdf"
OUTPUT_TAR="$REPO_ROOT/paper/dataset_paper_v6_arxiv.tar.gz"

BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/dsf-arxiv.XXXXXX")"
trap 'rm -rf "$BUILD_ROOT"' EXIT
STAGE="$BUILD_ROOT/submission"
mkdir -p "$STAGE/figs"

# arXiv receives paper.tex at the archive root, so remove the repository-only
# paper/ prefix from figure paths.
sed 's#paper/figs/#figs/#g' "$SOURCE_TEX" > "$STAGE/paper.tex"

cp "$REPO_ROOT/paper/cvpr.sty" "$STAGE/cvpr.sty"
cp "$REPO_ROOT/paper/silence.sty" "$STAGE/silence.sty"

for figure in dataset_anatomy_sqlite_pass failure_modes_64f reference_generation_pairs topology_metric_examples; do
  cp "$REPO_ROOT/paper/figs/${figure}.pdf" "$STAGE/figs/${figure}.pdf"
done

(
  cd "$STAGE"
  latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex >/dev/null
)

if rg -n 'LaTeX Warning: (Citation|Reference).*undefined|There were undefined references|multiply defined|Fatal error' \
  "$STAGE/paper.log"; then
  echo "arXiv compile contains unresolved references" >&2
  exit 1
fi

cp "$STAGE/paper.pdf" "$OUTPUT_PDF"
(
  cd "$STAGE"
  shasum -a 256 paper.tex cvpr.sty silence.sty figs/*.pdf > SHA256SUMS
  tar -czf "$OUTPUT_TAR" paper.tex cvpr.sty silence.sty figs SHA256SUMS
)

echo "$OUTPUT_PDF"
echo "$OUTPUT_TAR"
