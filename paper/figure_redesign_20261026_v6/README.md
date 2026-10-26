# TreeLoom figures, compact revision 6

[Prediction flow](TreeLoom_architecture.png) · [Encoder context](TreeLoom_encoder.png) · [Paper](../current_full_panel_working.pdf) · [Overleaf bundle](../current_full_panel_overleaf_working.zip)

This revision replaces the overly detailed tensor diagrams with two compact scientific figures. The overview follows one query leaf through three trees, node encoding, sequence assembly and ancestry prediction. The encoder expansion shows two shared tree rounds and the distinct roles of clade patterns and leaf memory. It omits routine activations, channel counts and low-level arithmetic.

The figures are 7 × 3.05 and 7 × 4 inches. The complete manuscript and a fresh extraction of its Overleaf source compile to 10 pages with identical extracted text. The full contact sheet and both figure pages were visually inspected. All eight result tables are unchanged and experiments remain stopped.

## Independent reading

The first reader recovered the main computation but found group selection unclear. The next version names age-cut groups, genomic distance decay and sharing across rounds and trees. A second fresh reader correctly recovered the main flow and both context mechanisms. Its remaining questions about exact interval features, age cuts and residual summation are clarified in the captions. This is a conceptual readability assessment, not a claim that the images specify every implementation detail.

- [First reader report](review_round1/reader_report.md) and [assessment](review_round1/verdict.md)
- [Second reader report](review_round2/reader_report.md) and [assessment](review_round2/verdict.md)
- [Source conformance](source_check.md)
- [Hash-bound reader acceptance](reader_acceptance.json)
- [Compilation and artifact receipt](integration.json)
- [Design rationale](review.md)

## Rebuild

The editable figure sources are `build_architecture.py` and `build_encoder.py`. Run them only within the dedicated project environment. `integrate.py` requires matching reviewed image hashes and checks that all result tables remain unchanged. `build_document.sh` compiles the manuscript and extracted Overleaf bundle, while `inspect_document.py` renders them for visual inspection. Generated build, render and log files remain local to the server repository and are excluded from the commit.

Model sources and the saved toy tree sequence were verified against the eight hashes in `source_hashes.json`. No training, simulation, profiling or queue operation ran for this redesign.
