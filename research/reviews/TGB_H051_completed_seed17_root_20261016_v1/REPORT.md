# Completed H051 causal trade diagnostic

Updated UTC: 2026-10-16T06:46:36.938241+00:00.

All three fixed seed17 models completed100epochs/100updates each,300totalupdates/6900TRAINgraphforwards plus90finalreplays. Genuine full-fit, VAL and TEST outer exits0 were consumed. All7heldout-year predictions were sealed before target access. No checkpoints, arms, normalization or hyperparameters were selected on TEST. Optional further seeds were declined using VAL while TEST remained closed.

| Method | VAL NDCG@10 ± year SD | TEST NDCG@10 ± year SD |
|---|---:|---:|
| native_persistence | 0.867241 ± 0.010687 | 0.846883 ± 0.014389 |
| first_order.seed17 | 0.857352 ± 0.007294 | 0.832384 ± 0.011250 |
| conditional_triple.seed17 | 0.856675 ± 0.005389 | 0.830455 ± 0.011007 |
| tree_only.seed17 | 0.858134 ± 0.005595 | 0.827057 ± 0.011339 |
| native_EMA_window7 | 0.842932 ± 0.008488 | 0.809007 ± 0.004883 |

Conditional-triple minus first-order: VAL -0.000677 ±0.002250, TEST -0.001929 ±0.000657 NDCG@10. These are descriptive sampleSDs of paired year differences. First-order exceeds tree-only on TEST by0.005327 ±0.000676, but all learned models are below native persistence on both splits. This condition does not support the higher-order memory mechanism or a new general operator. Do not rerun this unchanged answered condition.

VAL contains4seriallydependent years2010–2013/852country-years. TEST contains3seriallydependent years2014–2016/675country-years. SD is not optimizer-seed uncertainty. One seed, shared residual persistence prior and the forecast-before-current-year-edge protocol make this an initial causal utility diagnostic, not an exact published-leaderboard reproduction or confirmatory generalization result.

All output probabilities, final weights and raw observations stay on the server. Compact results and source provenance are published. Mean and sampleSD were independently recomputed from the written per-year scores without rerunning the numerical scorer or training.

[VAL results](../../../experiments/results/TGB_H051_CPU_VAL_explicit_evaluation_v1/evaluation_VAL.json) · [TEST results](../../../experiments/results/TGB_H051_CPU_TEST_explicit_evaluation_v1/evaluation_TEST.json) · [Completion lineage](completed.root_v1.json)
