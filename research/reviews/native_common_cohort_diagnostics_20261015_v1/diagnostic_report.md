# Native common-cohort diagnostics, 15 October 2026

The CPU computation completed for both original sealed native baselines on all eight internal VAL simulations. All AP C4/native6 and ARGMix raw seals passed before any new target access. The unchanged physical-span scorer was used for all seven original H038 cohorts. Sixteen separately recomputed FP32 primary accuracy/NLL cells exactly matched the prior baseline CPU receipts.

Mean ± sample SD below is over eight simulations (ddof1), with one fit per method. It is not optimizer-seed SD or the undefined ± in the ARGMix paper. Method-specific checkpoint selection was preserved. These are development results, and outer arrays were not read.

| Cohort | Native AP accuracy (%) | Current native ARGMix (%) | H038 aligned (%) | H038 zero concordance (%) |
|---|---:|---:|---:|---:|
| Present (182) | 93.78±0.84 | 94.45±1.15 | 95.49±0.83 | 94.93±0.79 |
| Bronze Age (162) | 94.76±1.18 | 97.71±0.75 | 98.17±0.80 | 98.41±0.54 |
| BAA (14) | 96.94±1.32 | 98.44±1.02 | 98.90±0.89 | 98.61±0.95 |
| Neo (302) | 97.57±0.27 | 99.30±0.26 | 99.41±0.13 | 99.17±0.27 |
| Yam (26) | 95.43±1.41 | 99.22±0.90 | 98.96±1.31 | 99.42±0.81 |
| Ancient (504) | 96.54±0.32 | 98.76±0.31 | 98.98±0.32 | 98.92±0.28 |
| All (686) | 95.81±0.45 | 97.61±0.50 | 98.05±0.43 | 97.86±0.37 |

Present-day boundary F1 at the frozen 1,000 bp tolerance:

- Native AncestralPaths: 7.03±0.78%.
- Current native ARGMix: 11.74±1.73%.
- H038 aligned: 13.42±1.47%.
- H038 zero concordance: 11.79±1.33%.

High base-pair accuracy does not imply accurate tract boundaries. Present-day boundary F1 remains low across all four methods. Yam boundary F1 is undefined in simulations with no true boundaries, so its full-eight mean/SD is left undefined; absent truth classes have undefined class recall.

Artifacts: native_cohort_metrics_mean_SD.json, per-parent compact metrics, CSV, same_support_comparison.json with paired accuracy differences and exact original-primary checks, and four PNG plots of accuracy/boundaries, confusion and reliability. No bulky per-query/tract/grid arrays were written. No retraining, new predictions, checkpoint changes, calibration, probability floor, normalization, GPU, admission, install or Git action occurred.

Recipe limitations are documented in docs/argmix_published_benchmark_audit_v1.md. Bronze_Age is only an approximate correspondence to published Post-Neolithic, and PFP remains unresolved.
