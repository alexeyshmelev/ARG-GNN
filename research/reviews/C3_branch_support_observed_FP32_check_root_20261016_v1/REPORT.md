# Bounded observed-tree feature check

Three deterministic 25%, 50% and 75% tree-index selections from American TRAIN parent 000 were evaluated once per implementation. All 1,600 observed haplotypes were retained. No ancestry truth or model checkpoint was read. No GPU or model training was used.

| Nodes | Sites | Original ms | Torch candidate ms | Maximum FP32 output difference |
|---:|---:|---:|---:|---:|
| 2614 | 2 | 12.376 | 16.808 | 1.49e-08 |
| 2636 | 4 | 18.697 | 8.118 | 2.98e-08 |
| 2627 | 1 | 8.779 | 7.575 | 0 |

All outputs passed the declared 2e-6 tolerance. Peak process RSS was 400 MiB. The original helper retains its existing NumPy FP64 accumulator and returns FP32. No FP64 model computation occurred.

Conclusion: integer-DP vectorization is plausible, but timing is mixed at the actual small SNP counts. This is a bounded engineering check, not general correctness or speedup evidence. The live implementation is unchanged. Do not repeat this same check or start a numerical-equivalence campaign. Current training-step optimization should target repeated hashing and CPU clones, which are a different stage.

Independent static review: [source verdict](../C3_actual_startup_CPU_feature_diagnosis_20261016_v1/VECTORIZATION_STATIC_REVIEW.md). Actual tool exit 0 is preserved in actual_tool_terminal.root_v1.json.
