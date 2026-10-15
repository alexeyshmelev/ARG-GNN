# Current full-panel AAAI working manuscript

This is the current scientific working draft, separate from the historical 140-diploid methods drafts. The research goal is not complete. It is not a submission-ready manuscript or a declaration of novelty or overall superiority.

Primary sources:

- current_full_panel_working.tex
- current_full_panel_refs.bib
- current_full_panel_provenance.json

The body uses the completed 476-diploid/952-haplotype, 32-TRAIN/eight-VAL H038, H040, H042, native AncestralPaths and current native ARGMix evidence. A compact H043 control appendix was added only after root confirmed completed evaluation. Its 0.0153 pp paired effect is below its registered 0.1 pp practical gate and does not displace the H042 main condition. H044 results are excluded.

The manuscript describes the actual H042 implementation: observed SNP pooling and known donor annotations, two shared full-tree PyG sweeps, node-local rank-8 gained/lost-clade residual, all 952 sampled contexts through the complete bidirectional GRU, and the 96→32→4 classifier. The TikZ architecture is embedded in the TeX source and requires no downloaded figure.

Tables trace to named evidence keys and exact source paths in the appendix. Full SHA256 values are in the provenance JSON. Accuracy SDs are across eight independent development simulations, ddof1, with one optimizer seed. The draft preserves the derived C4 endpoint, the native AP raw infinite NLL, method-specific training/selection, and weak tract-boundary performance. Numerical differences do not isolate architecture.

## Remaining scientific gates

- Independently assess the selected model on untouched simulation identities, and quantify optimizer-seed variability.
- Complete matched-input and matched-budget attribution, full-panel external comparators including ARGformer, and the separately verified original-paper benchmark protocols.
- Complete an independent current literature review before formulating a distinct architecture or operator claim.
- Establish robustness beyond the present demographic/observation condition, and obtain appropriate cross-domain and empirical biological evidence.
- Address transition recognition and readout resolution prospectively, preserving true short tracts and the frozen boundary scorer.

No original-paper reproduction, universal ranking, new-operator, statistical-significance, held-out-test, or real-data PCA claim is made. Sparse-cohort undefined metric counts remain in the original records.

## Root compilation and review

Compilation and rendering were explicitly left to root. No compiler, installation, Git operation, experimental array, checkpoint tensor, GPU, model execution, admission or accounting operation was used during authorship.

Use the existing remote project-local PDFTeX runtime and unchanged AAAI27 style/bibliography style, as described in paper/README.md. The native Mac standalone editor cannot access this remote multi-file project under the repository storage rule. No project file was written on the Mac.

The header follows the vendored official AnonymousSubmission2027 template with its internal draft option. It uses no forbidden layout package, hyperlink package, negative spacing, forced page break or modified style asset. TikZ statement delimiters are the only semicolons in the TeX source.

Target: 6–8 main pages, with bounded references/appendix and at most 20 pages total. Root compiled and visually reviewed all eight pages. The main text ends on page6, followed by references and compact appendices. There are no overfull boxes, unresolved citations or LaTeX warnings in the final compile. Static source analysis estimates 3,452 main-prose words, excluding tables and diagram. Root should compile/render, check figure text and arrows, table width/float placement, citations and page limits, then perform scientific review before any submission claim.

Static checks passed for balanced groups/environments, unique/complete labels, eight resolved citation keys, absence of unresolved placeholders, no prose semicolons, exact evidence hashes and untouched official style assets. This is source validation, not independent peer review.

## Frozen source references

- current_full_panel_working.tex SHA256 4165d4a455b0cb527b1600688149d86e0ae8580a09387371f7018a3e36549764
- current_full_panel_refs.bib SHA256 630d01e997364af760b7f9ac5d02be59f87e4e2b6db43a9d06be9601404c21af
- current_full_panel_provenance.json SHA256 d996f449c880c7d0365a9c44badc0a6c6ec06941a012b4378e4dfc6c66366870

Four inherited scholarly entries retain the existing verified publisher records. Relate and GRU metadata were verified through Crossref, PyG through DataCite, and msprime through Europe PMC bibliographic metadata after a Crossref 429 response. The msprime author list is explicitly abbreviated with “and others” in BibTeX. Verification URLs and extracted metadata are in the provenance JSON. No full text was downloaded.

Historical anchor/boundary/methods drafts and all experiment sources remain unchanged.

## Root artifact verification, 2026-10-15

Root compiled this multi-file project remotely with the existing project-local TeX runtime, rendered all8pages and visually inspected every page. The redundant bibliography-style declaration, boundary table width and architecture-label collision were corrected. Results and scientific claims were unchanged. Final compile has no overfull boxes, unresolved citations or LaTeX warnings. Remaining underfull-box notices arise chiefly from breakable provenance paths.

- [Current compiled working PDF](current_full_panel_working.pdf)
- [Overleaf working source bundle](current_full_panel_overleaf_working.zip)
- [Compile and visual-review receipt](build/current_full_panel_20261015_v1/compile_receipt.json)

This8-page development artifact preserves the outstanding scientific gates above. It is not a final publication. Root source SHA256: da5d85bb7e8fdf43fe545e10f2faf0e3a340de52fa9fa23da88838579ede351a. Updated provenance SHA256: 9c1c25bbb43b9d9fa8fe8c30ecdfa08ff514ae3c35ea5966c61701ff95d66612. Earlier author hashes record the pre-compilation draft.
