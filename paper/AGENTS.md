# Canonical dataset-paper release

The only active source of truth for the submitted dataset paper is:

- `dataset_paper_v6.tex`

Its canonical compiled output is:

- `dataset_paper_v6.pdf`

Files named `dataset_paper_v3*`, `dataset_paper_v4*`, and `dataset_paper_v5*`
are historical snapshots. Do not edit, compile, or treat them as current unless
Jin explicitly asks to recover an older draft.

Use `../scripts/build_arxiv_submission.sh dataset_paper_v6.tex` for the arXiv
source package. LaTeX build outputs belong under `paper/`, never at the
repository root.
