# H040 completed development comparison

Both FP32 arms finished 30 additional passes and 960 updates from the same H038 warm start. The frozen evaluation completed with actual launcher wait 0. All 96 prediction seals preceded VAL truth. One optimizer seed was used. SD is across eight VAL simulations.

| Arm | Present182 accuracy, mean ± SD (%) | NLL | Selected additional pass |
| --- | ---: | ---: | ---: |
| tensor | 96.2674 ± 0.7493 | 0.109332 | 30 |
| event_mlp | 96.2007 ± 0.7048 | 0.110014 | 30 |

The paired tensor-minus-event-MLP effect is 0.0667 pp, positive in 6/8 parents. This small development effect does not establish statistical significance or architectural novelty.

Native current-release ARGMix scores94.4461% on this endpoint. The tensor score is1.8212pp higher, but training/selection and observed SNP inputs differ. This is not the paper's1,611-diploid experiment, an architecture-only comparison or an independent test result.

| Population | Tensor accuracy, mean ± SD (%) | Event MLP accuracy, mean ± SD (%) |
| --- | ---: | ---: |
| age_stratum_BAA | 99.1439 ± 0.8901 | 99.1147 ± 0.9216 |
| age_stratum_Bronze_Age | 98.8266 ± 0.4696 | 98.8238 ± 0.4377 |
| age_stratum_Neo | 99.5200 ± 0.1579 | 99.5576 ± 0.1232 |
| age_stratum_Yam | 99.7215 ± 0.3045 | 99.7569 ± 0.2389 |
| all_queries_686 | 98.4933 ± 0.3390 | 98.4922 ± 0.2948 |
| primary_present_day_182 | 96.2674 ± 0.7493 | 96.2007 ± 0.7048 |
| secondary_ancient_504 | 99.2971 ± 0.2073 | 99.3197 ± 0.1707 |

Tensor does not outperform event MLP in every population. The complete primary/per-population values and paired effects are in [summary.json](summary.json). Raw confusion, calibration and boundary diagnostics remain in the original completion and selected diagnostic packets.
