# Frozen H045 application to the fixed Danish cohort

The completed fixed-model application gives later-minus-earlier EHG assignment +14.27 percentage points and CHG +5.52 points. These are descriptive neural assignments without ancestry truth or mixture calibration.

| Scope | EHG difference (pp) | CHG difference (pp) |
|---|---:|---:|
| chr20 | +14.21 | -0.76 |
| chr21 | +4.46 | +17.91 |
| chr22 | +24.13 | -0.59 |
| Covered-bp pooled | +14.27 | +5.52 |

The ten query people are the frozen five-earlier/five-later cohort. Both haplotypes are averaged per person after physical-span weighting, chromosomes are pooled by covered bp, then people are weighted equally within era. The same 18 individuals supply 36 haplotypes, with 16 reference and 20 query haplotypes. All 1,158 native trees and 8,587 qualified site records are retained. Features use original phased GT. Learned equations, normalizations and the original 31,812-parameter pass-30 H045 checkpoint are unchanged, with zero optimization, calibration or reselection.

EHG is positive on all three chromosomes and under all ten leave-person and three leave-chromosome deletions. CHG is positive under every leave-person deletion but is negative on chr20/22. Omitting chr21 gives CHG -0.676pp, so its pooled direction is chromosome 21-dependent. These sensitivity summaries are descriptive, not independent confidence intervals or p-values. The registration set no material-effect threshold.

The H055 GP-dosage affinity contrasts remain negative (-0.00223403 and -0.00460966). They use the same imputation resource, proxies and windows, with 13-person common support and SNP weighting rather than this ten-person span-weighted model output. They are counterevidence to simple biological corroboration and are neither independent truth nor the same estimand. A positive H056 assignment cannot rescue them.

Panel size36versus952haplotypes,10Mbversus2Mbwindows, and pre-size versus post-five-size Relate reconstruction differ from training. This is a descriptive out-of-distribution application, not an empirical local ancestry accuracy result or a calibrated population-history estimate. H056 was registered after H055 computation existed but before H056 outcomes were opened.

Actual application tool42438 returned0, consumed chunk1ba168. Original worker/launcher waits0, owned drain and final original scope guard succeeded. No files were deleted. Outer elapsed242.69s includes supervision/guards, with127.20s recorded scientific execution scope and732.11MB peak workerRSS. Those times are not an isolated model benchmark.

![Frozen model assignments and both registered contrasts](Danish_frozen_model_assignments.png)

[Scientific summary](../../../experiments/results/H056_frozen_H045_Danish_three_chromosome_CPU_v2/summary.json) · [Actual terminal evidence](../H056_three_bank_application_actual_root_20261016_v2/actual_tool_terminal.root_v1.json)
