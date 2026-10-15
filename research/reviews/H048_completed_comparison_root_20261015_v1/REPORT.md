# H048 graph-conditioned reset chain

H048 does not meet either registered utility rule and does not improve the best H045 development accuracy. Distinct operator novelty and independent-test performance remain unproved.

Both arms completed30FP32 passes/960updates from the sameH042 initialization. All96predictions were sealed before VALtruth. SD is across8reusedVAL simulations, one optimizer seed. This is development selection, not independent confirmation.

| Cohort | Arm | Accuracy % ± SD | NLL ± SD | Boundary F1 % ± SD |
|---|---|---:|---:|---:|
| primary_present_day_182 | graph_to_transition | 96.3800 ± 0.6002 | 0.3039 ± 0.0586 | 16.6695 ± 1.5616 |
| primary_present_day_182 | graph_to_own_confidence_fixed_distance_chain | 96.3886 ± 0.6098 | 0.1371 ± 0.0290 | 16.7423 ± 1.8672 |
| age_stratum_Bronze_Age | graph_to_transition | 98.8686 ± 0.4981 | 0.1002 ± 0.0511 | 15.3015 ± 3.1225 |
| age_stratum_Bronze_Age | graph_to_own_confidence_fixed_distance_chain | 98.8716 ± 0.4962 | 0.0494 ± 0.0251 | 19.5244 ± 2.6802 |
| age_stratum_BAA | graph_to_transition | 99.0791 ± 0.8563 | 0.0696 ± 0.0773 | 8.2341 ± 11.9081 |
| age_stratum_BAA | graph_to_own_confidence_fixed_distance_chain | 99.0846 ± 1.1561 | 0.0400 ± 0.0528 | 10.2083 ± 15.4416 |
| age_stratum_Neo | graph_to_transition | 99.4847 ± 0.2192 | 0.0455 ± 0.0200 | 22.3599 ± 2.1140 |
| age_stratum_Neo | graph_to_own_confidence_fixed_distance_chain | 99.6338 ± 0.0734 | 0.0165 ± 0.0054 | 24.7771 ± 4.4283 |
| age_stratum_Yam | graph_to_transition | 99.6233 ± 0.5314 | 0.0331 ± 0.0556 | 12.3263 ± 24.4074 |
| age_stratum_Yam | graph_to_own_confidence_fixed_distance_chain | 99.6818 ± 0.3024 | 0.0098 ± 0.0084 | 23.5622 ± 25.1925 |
| secondary_ancient_504 | graph_to_transition | 99.2826 ± 0.2593 | 0.0631 ± 0.0256 | 18.4662 ± 2.2462 |
| secondary_ancient_504 | graph_to_own_confidence_fixed_distance_chain | 99.3760 ± 0.1971 | 0.0274 ± 0.0103 | 22.2527 ± 2.5413 |
| all_queries_686 | graph_to_transition | 98.5125 ± 0.3421 | 0.1270 ± 0.0332 | 17.3526 ± 1.2309 |
| all_queries_686 | graph_to_own_confidence_fixed_distance_chain | 98.5834 ± 0.3006 | 0.0565 ± 0.0147 | 18.7646 ± 1.7659 |

Transition-minus-confidence accuracy: -0.008640pp, paired SD 0.144421pp. Boundary F1 delta: -0.072765pp.

Transition selectedpass5, own-confidence selectedpass20. Full11metrics across all7cohorts, actual scoring support and source/wait refs are in [summary.json](summary.json). No probability, checkpoint or query-truth payload was read for aggregation.
