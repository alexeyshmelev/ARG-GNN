# Scientific paper rewrite handoff

The rewrite is a nine-page scientific working draft. Main text ends on page 6, references span pages 6–7, and separately headed Supplementary Material occupies pages 8–9. Source, bibliography, an original editable vector schematic, official AAAI assets and compiled PDF are inside paper/rewrite_20261016_v1/.

The final remote build returned zero for PDFLaTeX, BibTeX, PDFLaTeX and PDFLaTeX. There are zero LaTeX warnings, overfull boxes or unresolved references/citations. All nine final pages were rendered and visually inspected. Figure 1 is page 3 and main score tables are pages 5–6.

## User-requested changes

- Rewrote the abstract as biological context, representation gap, method, supported result and implication.
- Rebuilt the narrative as introduction, related work, mathematical method, experimental design, results, discussion and conclusion.
- Removed library, driver, precision, path and runtime-audit prose and all E1–E18 path appendices.
- Replaced the old box chain with an original editable TikZ figure showing schematic ARG, ordered trees, persistent sample colors, allele/age/reference evidence, two shared tree sweeps, structural and sample-memory conditioning, RNN history, concatenation, classifier and ancestry output.
- Put source-cited simulation parameters, meaningful hyperparameters and extra scientific comparisons in the supplement.
- Bolded best means within comparable groups, including AP boundary recall and continuation boundary F1. Design tables and one-method source-class rows have no artificial competitive winner.
- Kept visible research TODOs and concise scientifically material comparison caveats.

Scores, trained models, experiment sources and prior current-paper files are unchanged. No new performance, top-1, independent-test, exact-reproduction or architectural-novelty claim is introduced.

## Competitor fidelity

The separate source-and-trace audit is complete at research/reviews/competitor_fidelity_user_audit_20261016_v1/source_handoff.json. Its material verified findings are incorporated:

1. Scored European ARGMix uses release 15 references and no ancestor nodes. The preprint specifies 25 references and three ancestors. The text explicitly labels the executed release condition. The supported public pipeline rejects ancestor inputs. Recovering the paper-era implementation, training specification and checkpoint is a visible TODO. A25+0 release sensitivity is separately named.
2. AP is the released dense six-path model with a derived C4 probability collapse. It realizes 5,190,874/5,200,000 requested rows. All 9,126 deficits are BAA/path6 across three simulations. Accuracy effect is untested. The original paper's convolutional model is distinguished from the archived dense model. Replacement sampling and32x2Mb versus3x200Mb genomic diversity are disclosed. Decreasing epoch5 training loss does not establish convergence.
3. ARGMix selects epoch 9 of 11 completed epochs under native patience 2 and cap 20, with all 32 TRAIN/eight VAL simulations.
4. No American ARGformer ancestry score is reported. The visible supplement TODO distinguishes paper temperature .05/inverse-frequency weighting from released .30/uniform fallback and requires completion of native pretraining, contrastive learning and retrieval.

Eight reused development simulations, one training seed, pipeline differences and the memory/boundary tradeoff remain explicit. The final audit establishes material setting differences and unresolved scientific TODOs. It is not a numerical-equivalence, matched-strength, convergence or intent certificate.

## Rebuild and integration

Run remotely from repository root:

.runtime/env/bin/python -B paper/rewrite_20261016_v1/compile_remote.py

Existing executables are .runtime/tex/.TinyTeX/bin/x86_64-linux/pdflatex and bibtex, plus /usr/bin/pdftoppm, pdftotext and pdfinfo. The build script records actual stage arguments and exit codes. AAAI style bytes, fonts and margins are unchanged. A standard fresh supplementary page was requested by root.

Portable assets: manuscript.tex, references.bib, aaai2027.sty, aaai2027.bst, figures/whole_tree_architecture.tex.

For promotion, root should copy the figure to paper/figures/whole_tree_architecture.tex and bibliography to paper/current_full_panel_refs.bib, then copy manuscript.tex to paper/current_full_panel_working.tex while changing only its bibliography basename from references to current_full_panel_refs. Recompile that integrated source and build the Overleaf zip from its actual bytes.

Final QA: build/final_rendered/page_01.png through page_09.png, and contact_1_3.jpg, contact_4_6.jpg, contact_7_9.jpg. Earlier build/rendered and final_rendered/contact_7_8.jpg are superseded and excluded from final QA.

The PDF skill marker was run exactly once before authoring. Its inspected implementation only validates arguments and performs no file writes. All manuscript, build, cache and rendered writes stayed in the assigned remote directory. No local project files, package installs, GPU jobs, admissions, accounting operations, commits or pushes were made by this worker.

## Scientific work still needed

Complete common-data American comparisons, deliberately reconstruct original ARGMix settings, evaluate untouched European simulations, establish a useful independently assessed architectural contribution, finish a justified native GST cross-domain comparison and obtain a final fresh-context scientific review. Danish model assignments remain descriptive. The current trade diagnostic is negative.

## Final bounded audit integration

The prior-manuscript evidence now points to root's preserved archive research/reviews/paper_root_integration_20261016_v1/prior_current_full_panel_working.tex, SHA3cee74e244366c50ae56fdf6225d629d8c07b47f59e7a632cbf164baa0e03e84. This avoids binding historical evidence to the mutable current manuscript path. The final source package is sealed in source_handoff.json. Scientific benchmarks and novelty remain open.
