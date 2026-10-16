# Scientific manuscript rewrite and benchmark status

The nine-page AAAI working draft was rewritten around the biological problem, field gap, related work, conceptual and mathematical methods, and measured findings. The main text omits software/runtime trivia. An original editable ARG-to-ancestry schematic replaces the prior figure. The supplement covers scientific simulation parameters and sources, training schedules and comparisons. E1-E18 evidence-path appendices were removed. Best comparable table entries are bold, including baseline wins. Missing material author-method details are visible Research TODOs.

Both the integrated LaTeX and independently extracted Overleaf sources compiled cleanly. All nine final pages were rendered and visually inspected. This editorial revision does not change any experiment, score or baseline.

| European shared development pipeline | Present182 span accuracy, mean ± simulation SD (%) |
|---|---:|
| Stable-leaf candidate | **97.0778 ± 0.5674** |
| Same-budget continuation | 97.0000 ± 0.4978 |
| Current author ARGMix release | 94.4461 ± 1.1491 |
| Released AncestralPaths, derived C4 | 93.7830 ± 0.8385 |

Eight reused 2 Mb development simulations, one optimizer seed. The stable-leaf minus continuation difference is 0.0778 percentage points, with lower boundary F1, 18.3807% versus 19.0488%. This does not establish useful higher-order novelty. Current ARGMix uses 15 references and no ancestors, compared with the preprint's 25 references and three ancestors. The public release rejects ancestor inputs. The released AncestralPaths dense model differs from the published convolutional description. Its six-to-four-source collapse is a derived endpoint. Input, exposure and model-selection differences prevent architecture-only or original-table reproduction claims.

American C3 training has at least 336/600 completed optimizer updates recorded, with checkpoint200 independently verified. No ancestry score yet. Native FLARE on one American VAL000 has 98.0100% focal accuracy and 97.7151% macro F1 over 10,000 fixed focal records, without a multi-parent SD or completed GNN/ARGformer head-to-head. Native ARGformer initialized a 418 MiB CUDA context then failed a controller-registry freshness check, with verified worker -15/outer241 and owned drain. Optimizer progress remains unverified. Its same-charge operational amendment is under independent review.

Full 1,611-diploid/3,222-haplotype Relate inference is running over all 311,599 observed SNPs and the full 35,178,458 bp map. No inferred-bank completion or ancestry score is claimed. Eight reserved European outer simulations remain ungenerated and untouched. Their source-only preparation workflow is under independent review. The paper remains a working draft, not a completed submission or an overall top-1 result.

[Current PDF](../../../paper/current_full_panel_working.pdf) · [Overleaf bundle](../../../paper/current_full_panel_overleaf_working.zip) · [Competitor-fidelity matrix](../competitor_fidelity_user_audit_20261016_v1/comparison_matrix.md) · [Build and visual check](../paper_root_integration_20261016_v1/integration_build_receipt.root_v1.json) · [Detailed current scores](../../current_scoreboard.md)
