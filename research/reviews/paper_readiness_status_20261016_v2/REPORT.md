# Paper readiness metadata audit, 16 October 2026

There is real completed development progress and a compiled eight-page AAAI draft. The required benchmarks and the intended distinct, useful architecture contribution remain unfinished. The project is not submission-ready and no overall top-1 claim is supported. This assessment reads source/report and compact score metadata only.

| Completed shared European development pipeline | Present182 span accuracy (%) | NLL | Boundary F1 at1kb (%) |
|---|---:|---:|---:|
| H045 foreign episode | 96.8270 ± 0.5870 | 0.092980 | 18.1558 ± 1.3790 |
| Matched H045 own state | 96.8175 ± 0.6142 | 0.092976 | 18.0557 ± 1.7761 |
| Current native ARGMix | 94.4461 ± 1.1491 | 0.219695 | 11.7372 ± 1.7328 |
| Released AncestralPaths, derived C4 | 93.7830 ± 0.8385 | infinite raw FP32 | 7.0281 ± 0.7779 |

Scope: 32 TRAIN/eight reused VAL simulations,2Mb,952observed haplotypes,686query haplotypes,one optimizer seed. SD is across VAL simulations. Foreign episode exceeds current ARGMix by2.3809pp and derived-C4 AP by3.0441pp in this pipeline condition. Different SNP evidence, training and selection prevent architecture-only or published-table claims. Its gain over the matched own-state arm is only0.009568pp, below the0.10pp practical gate. All-query accuracy is slightly higher for own-state,98.7815% versus98.7761%.

The scientifically material gaps are:

1. **Useful novelty:** The matched episode mechanism does not establish useful foreign-state exchange. Independent literature audits identify established conditional-mean/higher-order constructions. New candidates need decisive matched evidence and a narrower defensible novelty claim.
2. **External/native ancestry comparisons:** Shared European ARGMix/AP scores exist. The full American30Mb ARGformer/FLARE/our-GNN ancestry table does not. The inspected scoreboard has no common scores. The native ARGformer retrieval route remains a disclosed reconstruction, with paper PCA10 clustering details unavailable.
3. **Paper-scale European evidence:** The prospective1611-diploid paired comparison is planned but not simulated/trained/scored. Current2Mb development is not original-table reproduction.
4. **Independent confirmation and tract quality:** Eight reserved outer simulations and optimizer-seed variability are unfinished. H045 detects only22.59% of true present-day changes at1kb, despite96.83% span accuracy.
5. **Empirical biological result:** H045 probabilities/proportions/PCA exist for24haplotypes/12subjects and are descriptive. The Danish GP contrasts support neither registered positive direction. The corresponding frozen H056 three-window model application is unfinished.
6. **GST/cross-domain efficacy:** Four author GST modes completed a TRAIN1000/VALID1495 seed0 pilot, best AP0.65378329. Our molecular model and paired TEST are absent. GST inputs are unordered reactant sets. TGB causal baseline scores exist, but our learned model has no score.
7. **Final paper:** Compilation/layout is done for the current development draft. Final selection, stronger evidence, updated architecture/results and fresh independent peer review remain.

The newer H056 compact status records successful recovered chr20 acquisition/bridge/readback. README still describes that recovery as future. This is an engineering/data-readiness advance, without a new ancestry score. Live execution was not probed in this metadata task.

Exact source/receipt hashes, every H045/native cohort accuracy and completion scope are in [report.v2.json](report.v2.json). No observation, truth, checkpoint, bank, embedding or prediction array was opened, no model or scorer was executed, and no jobs, admissions or Git actions occurred.
