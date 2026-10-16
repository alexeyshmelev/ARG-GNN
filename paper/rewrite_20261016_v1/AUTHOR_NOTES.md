# Scientific rewrite, October 16, 2026

This is a human-facing scientific working manuscript. The experiments and recorded metrics are unchanged. It is not a submission-ready paper or a claim of architectural novelty or benchmark-wide superiority.

- Main manuscript: manuscript.tex
- Bibliography: references.bib
- Editable original architecture: figures/whole_tree_architecture.tex
- Compiled PDF: manuscript.pdf
- Original official styles: aaai2027.sty and aaai2027.bst
- Remote rebuild: compile_remote.py

The one-paragraph abstract follows biological context, representation gap, proposed approach, completed results and their implication. The introduction motivates whole-tree structure and sampled-haplotype correspondence. Related work distinguishes AncestralPaths, ARGMix, ARGformer, GST and temporal graph recurrence. Main methods explain actual information flow and equations, without software, device, precision or execution-audit detail. Supplementary Material contains scientific simulation parameters, meaningful model/training settings, additional cohort/ablation results and the initial cross-domain diagnostic.

Best means are bold within each comparable table group. AP's boundary recall and the continuation control's boundary F1 are correctly highlighted. Demographic/budget tables and the single-method per-source FLARE table do not imply a competitive ranking.

Important scientific completion TODOs remain visible: untouched European testing, native American comparison, source-cited original-paper simulation/evaluation settings, ARGMix 25-reference/3-ancestor paper versus evaluated 15-reference/no-ancestor release, ARGformer paper versus release temperature/anchor weighting, and a justified GST comparison.

The main model schematic depicts the scored stable-leaf configuration, not unscored H058. The generic ARG and trees are original illustrative vector drawings. Repeated colors denote stable sampled haplotypes, not asserted internal-ancestor identity.

All E1-E18 path-listing appendices are removed. Exact traceability is preserved in the separate research handoff, outside the paper.

The completed competitor source-and-trace audit is incorporated. Original ARGMix25+3 requires recovery of the paper-era implementation, training specification and checkpoint. A25+0 current-release sensitivity is separate. The original AP convolutional model, archived dense implementation, bounded sampler, genomic diversity and convergence uncertainty remain distinguished.
