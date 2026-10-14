# Development cohort metrics

Mean ± sample standard deviation across eight independent validation simulations. One training seed. These are our 476-diploid, 2 Mb development data, not the ARGMix published benchmark. Original cohort labels are retained until the published roster mapping is audited.

| Cohort | Haplotypes | Accuracy (%) | Macro F1 (%) | Boundary F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| age_stratum_BAA | 14 | 98.90 ± 0.89 | 97.60 ± 2.67 | 11.45 ± 13.73 |
| age_stratum_Bronze_Age | 162 | 98.17 ± 0.80 | 98.07 ± 0.70 | 13.42 ± 3.92 |
| age_stratum_Neo | 302 | 99.41 ± 0.13 | 99.31 ± 0.13 | 18.35 ± 3.61 |
| age_stratum_Yam | 26 | 98.96 ± 1.31 | 99.41 ± 0.74 | N/A |
| all_queries_686 | 686 | 98.05 ± 0.43 | 97.69 ± 0.48 | 14.20 ± 1.40 |
| primary_present_day_182 | 182 | 95.49 ± 0.83 | 95.34 ± 0.73 | 13.42 ± 1.47 |
| secondary_ancient_504 | 504 | 98.98 ± 0.32 | 98.69 ± 0.44 | 15.59 ± 2.89 |

Boundary scores use the existing 1 kb tolerance. High per-base accuracy does not guarantee reliable tract boundaries. Complete definitions and confusion matrices are in our_cohort_metrics_mean_SD.json.
