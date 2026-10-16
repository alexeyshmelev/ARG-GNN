# Paper and benchmark completion audit

**The nine-page AAAI development draft compiles and has an Overleaf bundle. The scientific paper and required comparisons remain incomplete.** Architecture equations, a schematic, development tables, provenance and the Danish application are written.

| Completed European pipeline | Present182 accuracy (%) | NLL | Boundary F1 (%) |
|---|---:|---:|---:|
| H045 foreign episode | 96.8270 ± 0.5870 | 0.092980 | 18.1558 ± 1.3790 |
| Matched own-state control | 96.8175 ± 0.6142 | 0.092976 | 18.0557 ± 1.7761 |
| Current native ARGMix | 94.4461 ± 1.1491 | 0.219695 | 11.7372 ± 1.7328 |
| Released AncestralPaths, derived C4 | 93.7830 ± 0.8385 | infinite | 7.0281 ± 0.7779 |

Scope: 32 TRAIN/eight reused VAL simulations, 2 Mb, 952 haplotypes, 686 queries, one optimizer seed. SD is across simulations. The +2.3809 pp ARGMix difference includes input, training and selection effects. The matched episode gain is only +0.009568 pp, below its +0.10 pp gate.

| Evidence | Completed | Pending |
|---|---|---|
| ARGMix | Author-code European comparison, seven cohorts and broader metrics | Prospective 1,611-diploid simulation/fitting/scoring. Exact published recipe unavailable |
| AncestralPaths | Native Dense512/five epochs, six-path outputs collapsed to declared C4 | Independent confirmation, not complete Nature-study reproduction |
| ARGformer | Distinct historical adapted pilot, American observations/native corpus | Native per-target training/retrieval and common EUR/EAS/AFR scores |
| FLARE | Author build and TRAIN/VAL000 predictions prepared | Common held-out ancestry table and remaining targets |
| GST | Four author modes, TRAIN1000/VALID1495, seed0, best VALID AP 0.653783. Full TRAIN parsing 10,439 reactions | Full-data fits, our molecular adapter and paired TEST |
| Untouched ancestry tests | Eight outer simulations reserved | Final-method evaluation and optimizer-seed uncertainty |
| Real genomes | Frozen H045, ten Danish people, three windows, 1,158 trees, 8,587 sites, zero optimization | No ancestry truth. EHG +14.27 pp is chromosome-consistent, CHG +5.52 pp reverses without chr21. Genotype contrasts support neither registered direction |
| Useful novelty | Whole-tree PyG/genomic processing and matched controls implemented | Distinct useful mechanism unproved. TGB triple TEST 0.83045 trails first-order 0.83238 and persistence 0.84688 |

H050 completed 960 new FP32 updates per arm, 3,840 total. Original TRAIN CLI exit 1 is preserved. Its ancestry evaluation awaits verified completion and consumed CPU outer exit 0 in the inspected reporting snapshot. No H050 scored output was opened.

Scientific completion requires matched American comparisons, the larger ARGMix study, a useful mechanism gain, untouched ancestry tests, final integration and fresh independent paper review. **Overall top-1 is unestablished.**

Basis: current_full_panel_working.tex SHA2cf643829495667fcea88a670181ab59cbdd3f192feffadbb4a87aa755b8ea04, paper/README.md, current_scoreboard.json snapshot 2026-10-16T07:04:06Z, progress_current.md and H056 current status. Older readiness REPORT.v3 is stale about page count, H056 and learned TGB completion. No jobs, scoring, probes, Git or reporter edits occurred.
