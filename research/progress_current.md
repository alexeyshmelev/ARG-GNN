# Research progress

Updated UTC: 2026-10-16T06:04:26.942560+00:00.

The compiled nine-page AAAI draft and Overleaf working bundle are available. The paper is not submission-ready. The main unfinished work is evidence for a useful distinct contribution and complete faithful comparisons.

| Shared European development pipeline | Present182 span accuracy, mean ± simulation SD (%) |
|---|---:|
| Our best completed H045 | 96.8270 ± 0.5870 |
| Matched H045 own-state control | 96.8175 ± 0.6142 |
| Current author ARGMix release | 94.4461 ± 1.1491 |
| Released AncestralPaths, derived four-source endpoint | 93.7830 ± 0.8385 |

All scores use eight reused VAL simulations, one optimizer seed, a 2 Mb shared panel. Differences in SNP inputs, training and selection prevent an architecture-only comparison or an original-paper reproduction claim. The best candidate gains only 0.0096 percentage points over the matched control. Its boundary F1 is 18.16%, so high span accuracy does not establish precise ancestry tract recovery.

## Current execution

- H050 European four-arm training: 27/30 complete TRAIN passes, 864/960 verified updates per arm. No validation score yet.
- American C3 GNN: running on GPU 0. The first saved checkpoint verifies 100/600 updates. Progress after that checkpoint is unknown.
- Native ARGformer: held and never started on this evaluation panel. FLARE predictions exist, but the common American ancestry table is unfinished.
- TGB three-arm full CPU campaign: launched and exact worker live, 100 epochs per arm planned. No completed optimizer milestone or learned score yet verified.
- GPU 0 snapshot: 36% utilization and 28047 MiB allocated. This is an instantaneous reading.

## Evidence still needed

1. Common American ARGformer, FLARE and GNN ancestry scores with the disclosed author-code reconstruction.
2. Larger prospective ARGMix comparison. The 1,611-diploid panel has not been simulated or scored, and exact original-table reproduction remains unavailable.
3. Eight untouched European outer tests and training-seed uncertainty for the final selected method.
4. A useful matched gain attributable to the proposed mechanism and a defensible novelty claim.
5. Completed learned cross-domain evaluation and stronger biological corroboration. The completed Danish application is descriptive. Its CHG contrast is chromosome-sensitive, and the independent genotype contrasts did not support the registered directions.
6. Final result tables, architecture figure and independent manuscript review after the scientific evidence is complete.

## ARGMix correction

The modern documented `build-training` route at author commit `c2481248eb0de5fda659d68f59c51e0e27becd20` uses `FastSubgraphEngine`, whose reference ranking uses uncapped absolute differences of quantized times. The earlier log-distance description applies to a legacy route. The old proposed shared-only adapter is a development adaptation and must not be run or described as author-faithful. The additive modern adapter is being corrected before execution.

[Full current scores](current_scoreboard.md) · [Working PDF](../paper/current_full_panel_working.pdf) · [Overleaf working bundle](../paper/current_full_panel_overleaf_working.zip)
