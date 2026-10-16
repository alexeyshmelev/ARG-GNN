# Research progress

Updated UTC: 2026-10-16T09:54:17.584056+00:00.

The compiled eleven-page AAAI draft and Overleaf working bundle are available. The paper is not submission-ready. The main unfinished work is evidence for a useful distinct contribution and complete faithful comparisons.

| Shared European development pipeline | Present182 span accuracy, mean ± simulation SD (%) |
|---|---:|
| H050 stable-leaf memory, current accuracy leader | 97.0778 ± 0.5674 |
| H050 same-budget continued training | 97.0000 ± 0.4978 |
| Historical H045 best | 96.8270 ± 0.5870 |
| Matched H045 own-state control | 96.8175 ± 0.6142 |
| Current author ARGMix release | 94.4461 ± 1.1491 |
| Released AncestralPaths, derived four-source endpoint | 93.7830 ± 0.8385 |

All scores use eight reused VAL simulations, one optimizer seed, a 2 Mb shared panel. Differences in SNP inputs, training and selection prevent an architecture-only comparison or an original-paper reproduction claim. H050 stableleaf adds0.0778percentagepoints over same-budget continuation but lowers boundaryF1 from19.0488%to18.3807%. Triple-support failed both registered utility rules,so higher-order novelty remains unsupported.


**First American native baseline:** FLARE reaches **98.01% focal accuracy** and **97.7151% macro F1** on the single VAL000 simulation, all10,000 fixed common focal records. F1: EUR97.50%, EAS98.5822%, AFR97.0629%. No between-parent SD or GNN/ARGformer head-to-head score exists yet. [Full metrics and scope](reviews/FLARE_VAL000_score_H020980_root_20261016_v1/REPORT.md).

## Current execution

- H050 European four-arm training and saved-fit evaluation completed. Stableleaf97.0778%,triple97.0640%,marginal97.0504%,continuation97.0000%. All23setups/sevencohorts and SDs are in the scoreboard. Both triple-support utility criteria failed.
- American C3 GNN: C3 is training on physical GPU 0 after all 20 TRAIN observation banks completed. At least 147/600 optimizer updates are verified, including 47 after retained checkpoint 100. Latest progress receipts do not assert a new durable optimizer checkpoint. No American GNN ancestry score.
- Native ARGformer: Native American ARGformer has no completed optimizer update or ancestry score. The prior GPU0 route failed before training because of inherited GPU eligibility checks. A fresh operational successor fixes the source route and is under independent review. It will run after the current C3 controller finishes, within GPU0 memory limits. Author numerical training and the original failure remain unchanged.
- TGB full campaign and VAL/TEST evaluation completed. TEST NDCG@10: first-order0.83238±0.01125, conditional-triple0.83045±0.01101, tree-only0.82706±0.01134, persistence0.84688±0.01439. No higher-order benefit. SD across3dependent years, one optimizerseed.
- GPU0 has an unrelated13542MiB process. Our C3 continuation is running, native monitor correction proceeds separately. Device utilization must not be attributed wholly to our project.

## Evidence still needed

1. Complete the common American ARGformer, FLARE and GNN ancestry table across original validation parents. First native FLARE VAL000 score is complete, GNN/ARGformer and remaining parents are pending.
2. Larger prospective ARGMix comparison. The 1,611-diploid panel has not been simulated or scored, and exact original-table reproduction remains unavailable.
3. Eight untouched European outer tests and training-seed uncertainty for the final selected method.
4. A useful matched gain attributable to the proposed mechanism and a defensible novelty claim.
5. Useful learned cross-domain evidence and stronger biological corroboration. The completed Danish application is descriptive. Its CHG contrast is chromosome-sensitive, and the independent genotype contrasts did not support the registered directions.
6. Final result tables, architecture figure and independent manuscript review after the scientific evidence is complete.

## ARGMix correction

The modern documented `build-training` route at author commit `c2481248eb0de5fda659d68f59c51e0e27becd20` uses `FastSubgraphEngine`, whose reference ranking uses uncapped absolute differences of quantized times. The earlier log-distance description applies to a legacy route. The old proposed shared-only adapter is a development adaptation and must not be run or described as author-faithful. The modern native training and inference adapters passed independent source review. Simulation, fitting and scoring remain pending. Native ancient-query self/homolog eligibility must be disclosed and aligned deliberately.

[Full current scores](current_scoreboard.md) · [Working PDF](../paper/current_full_panel_working.pdf) · [Overleaf working bundle](../paper/current_full_panel_overleaf_working.zip)

[Completed TGB result](reviews/TGB_H051_completed_seed17_root_20261016_v1/REPORT.md). The1GBqualificationreservation was closed with genuinewait0 and targetedwriterabsence, retaining188416bytes and crediting999811584unusedbytes. No files were deleted and no guard was changed.

[Current paper and benchmark completion audit](reviews/paper_completion_compact_audit_20261016_v1/REPORT.md). The eleven-page draft has equations and a schematic, but native American comparisons, the larger ARGMix panel, useful novelty and untouched tests remain unfinished.

Capacity recovery: one old completed run retained23.379GB of its52.321GB cap. Actual unused reservation credit28.942GB, no files deleted. Future reservations use the reviewed H020v3 guard. Existing C3/controller inputs and caps remain unchanged. [Actual recovery](reviews/prospective980_H020_root_activation_20261016_v3/REPORT.md).
