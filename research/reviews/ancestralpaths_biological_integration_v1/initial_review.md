# AncestralPaths biological integration v1: initial audit

Status: blocked by two concrete integration issues in runner SHA256
`53d6f871fef1d20fe865c82a9b2f40afef4b5fbbfd8f448d09ddb3c791f1103e`.
This is a static audit against the finalized Relate producer. No runner/author
code executed, no truth arrays or inferred trees opened, and no installs, GPU,
Git, or runner modifications performed. Exact source/roster/file hashes are in
`initial_review.json`.

## Blockers

1. **Manifest/roster contract (runner lines137-165; docs25-48).** The runner
   expects `input_kind=relate_inferred`, passed/completed status and older fields.
   Producer uses `kind=biological_paths_relate_v1`, final status
   `four_development_parents_inferred_pre_and_post_five_iterations`, `replicates`,
   parent status `inferred_both_conditions`, and `inferred_tree`,
   `observed_sample_manifest`, `completion_provenance`, and all280 explicit
   sample-index/node mapping objects. The derived actual-ID roster necessarily
   differs in SHA from the original source roster. Bind its source reference and
   observed attributes, then validate/use actual inferred IDs.
2. **Local coordinate and score support (runner188,202,324,444; docs52-58).**
   Producer trees are local `[0,coverage_right-coverage_left)`, with
   physical_original=local+coverage_left. Features need center-offset; TRAIN
   label joins need site+offset. Only declared retained center indexes may be
   evaluated. VAL exact scoring must intersect their physical100bp bins with
   observed coverage and truth intervals, exclude tails, and adapt tracts/
   censoring to scored support. Keep original grid indexes and2Mb normalization.
   The current full2Mb length check rejects valid trees; simply removing it
   would leave shifted labels/features and extrapolated tail scores.

## Checked invariants

All17 author file hashes match the frozen pin; training.py SHA is
`13b4f30ab0df75fc461b764ae2cdbd1fdc071ff45200e6401e84faa565508e4f`.
The TRAIN sampler follows source31-35,50-156: per-parent targets162,500, initial
floor stride with10,000/3 overhead, ordered site/query scan, cumulative caps
6,250 BAA/12,500 others, and repeated replacement with float step*=1.5.
The128-pass/no-progress failures are disclosed adaptations; no cap relaxation
or class deletion exists. Native Dense architecture/loss/5epochs are preserved.

VAL raw six-path/C4 predictions are written and hashed before the VAL truth
capability is opened. There is no VAL fit/checkpoint/early-stop/calibration
selection. C4 sums `(p0+p4,p1+p5,p2,p3)` before argmax. Confusion axes and
no-floor/infinite-NLL handling are correct. Existing intersection code integrates
within-bin truth switches exactly for its full-domain assumption; B2 remains.

All four actual observed rosters have96 pure refs (24 each Ana0,CHG1,WHG2,EHG3),
64 mixed refs with null classes, and120 queries with null classes. Queries are
present64,BronzeAge16,BAA8,Neo16,Yam16. Native feature shells appropriately count
declared ancient cohorts, including mixed refs and ancient focal self-inclusion;
query roles control supervised/predicted rows. No additional source sampler,
C4 probability, VAL-selection or pure-reference alignment blocker was found.

Runner owner received both blockers and will supply a revised hash for re-audit.
Runtime correctness and actual optimizer fitting remain unverified by this audit.
