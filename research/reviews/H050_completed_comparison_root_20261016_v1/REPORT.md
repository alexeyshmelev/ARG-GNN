# H050 completed matched comparison

All four arms completed 30 passes and 960 new FP32 updates each, 3,840 total. CPU evaluation completed all 192 prediction products before truth and returned genuine worker/outer exit 0. Original TRAIN CLI exit 1 remains preserved.

| Arm | Present182 accuracy % ± SD | Boundary F1 % ± SD | NLL | Selected new pass |
|---|---:|---:|---:|---:|
| stable_leaf_memory | 97.077844 ± 0.567395 | 18.380657 ± 1.314072 | 0.08811505 | 30 |
| triple_support_pair_memory | 97.064043 ± 0.559495 | 18.926459 ± 1.418678 | 0.08847067 | 30 |
| adjacent_marginal_null | 97.050364 ± 0.579588 | 18.849846 ± 1.052004 | 0.08839387 | 30 |
| h045_continued_training | 97.000028 ± 0.497841 | 19.048847 ± 1.427489 | 0.08854074 | 20 |

Stable-leaf memory raises the historical best development accuracy from 96.8270% to 97.077844%, a +0.250844 percentage-point difference that includes additional training and a different module. The same-budget continuation reaches 97.000028%. Simple memory therefore adds 0.077816 points while reducing boundary F1 by 0.668189 points. Triple-support does not beat stable-leaf accuracy. Both preregistered utility rules fail. A higher-order contribution and overall top-1 remain unestablished.

Scope: 32 TRAIN and eight reused 2 Mb VAL simulations, 952 haplotypes and 686 queries, one optimizer seed 101. SD is across simulations. Selected checkpoint lineage includes 2,880 earlier model updates plus 960 new updates per memory arm, or 640 new updates for selected continuation pass 20. Newly executed training totals 3,840 updates across the four arms.

[All23models/sevencohorts/calibration/boundaries](../../current_scoreboard.md) · [Independent arithmetic audit](../H050_completed_comparison_independent_20261016_v1/REPORT.md) · [Actual completed evaluation](../../../experiments/results/H050_CPU_full192_savedfit_failedCLI_root_v1/completed.json)
