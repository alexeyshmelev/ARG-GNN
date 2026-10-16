# Completed H057 TRAIN resolution diagnostic

This is a truth-oracle representation diagnostic on the 32 original TRAIN simulations. It is not a learned ancestry accuracy or a held-out result. All seven original cohorts and the original clipped 100 bp center scored support are retained.

| Cohort | Native-hold oracle accuracy (%) | 100bp-grid oracle accuracy (%) | Grid minus hold (pp) |
|---|---:|---:|---:|
| Present182 | 99.892032 ± 0.009986 | 99.996475 ± 0.000208 | 0.104443 ± 0.009838 |
| Bronze_Age162 | 99.969755 ± 0.003367 | 99.998996 ± 0.000087 | 0.029241 ± 0.003336 |
| BAA14 | 99.995181 ± 0.005669 | 99.999839 ± 0.000155 | 0.004658 ± 0.005547 |
| Neo302 | 99.979360 ± 0.002430 | 99.999319 ± 0.000061 | 0.019959 ± 0.002406 |
| Yam26 | 99.995768 ± 0.002629 | 99.999860 ± 0.000090 | 0.004092 ± 0.002569 |
| ancient504 | 99.977559 ± 0.001933 | 99.999257 ± 0.000047 | 0.021699 ± 0.001907 |
| all686 | 99.954868 ± 0.003338 | 99.998519 ± 0.000067 | 0.043651 ± 0.003301 |

SD is the sample SD across 32 independent TRAIN simulations. Equal-parent summaries and separately labelled pooled masses/counts are available in the original result. Native masses retain the unchanged FP32 helper, with float64 scalar sums. Grid intersections use original float64 coordinate arithmetic. No additional precision fixture or native mass recomputation was performed.

The registered primary gate was at least +0.10 percentage points on Present182. The observed +0.104443 pp passes narrowly. It establishes limited representational headroom for a finer readout. It does not establish an achievable gain, model superiority, novelty or a causal explanation of current model errors. Native and grid partitions are the original scorer readouts over the same physical support, rather than arbitrary continuous native-tree boundaries.

The proposed marker/tree feedback model remains conditional on the H050 validation result and a useful evidence-learning mechanism. Do not justify a large ancestry gain by the resolution ceiling alone. No new architecture is registered or trained by this audit.

Actual root tool 96442 consumed exit 0, chunk 0ce8fd. Worker wait 0 and owned drain are verified. The outer monitor recorded279.952seconds including admission before final completion guards and peak aggregate RSS550764544bytes. All science outputs fit the 2 MiB reservation. All old files are preserved.

[Original complete result](../../../experiments/results/TRAIN_native_hold_resolution_audit_root_v1/result.json) · [Actual terminal](actual_tool_terminal.root_v1.json) · [Registered protocol](../native_hold_resolution_train_registration_root_20261016_v1/protocol.root_frozen_v2.json)
