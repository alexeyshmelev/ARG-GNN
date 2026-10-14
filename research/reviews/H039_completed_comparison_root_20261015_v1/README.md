# Completed H039 development comparison

All three arms completed ten additional passes and 320 full-parent updates each from the same aligned H038 checkpoint, with reset AdamW and FP32 seed101. Actual TRAIN and evaluator launchers both exited0. Pass10 is selected for every arm by the prospectively frozen Present182 accuracy rule over passes5/10.

| Arm | Accuracy, mean ± sample SD | NLL |
| --- | ---: | ---: |
| directed | 95.712 ± 0.968% | 0.125775 |
| plain_continuation | 95.564 ± 1.041% | 0.130260 |
| generic_unary | 95.496 ± 1.064% | 0.132449 |

SD describes eight simulation parents. It does not describe training seeds. Directed improves over plain continuation in seven of eight parents, with a mean gain of0.1484 percentage points. This is a validation-selected development result, not original-paper reproduction or independent final-test superiority. The gain does not establish architectural priority or biological tract-boundary utility.

See summary.json for paired differences, selected checkpoint/source hashes and actual wait evidence. Historical failed attempts are retained and do not count as successful scientific runs.
