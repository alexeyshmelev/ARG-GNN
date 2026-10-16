# H058 observed-hold FP32 qualification

Actual root tool43084 exited0. The owned worker exited0, all owned processes/groups drained, and no operational failure was reported.

The original arithmetic protocol passed 37 checks on the first eligible TRAIN000 hold:1,903 nodes,4SNPs,952observed haplotypes,686queries and266references. FP32 forward/oracle,gradient,mask,permutation,null and zero-adapter checks passed at original tolerances. Numerical time364.25seconds, supervisor time405.51seconds, sampled combined peak RSS946,110,464bytes. No FP64,no optimization,no CUDA,no ancestry labels/checkpoints/VAL/TEST reads.

The original scalar-versus-dict FAIL is preserved. This corrected interface execution is a separately registered source correction. It establishes arithmetic behavior on one observed hold, without full-parent memory/throughput feasibility or ancestry accuracy. The complete32-parent/960-update-per-arm training experiment remains pending runtime admission.

[Actual receipt](../../../experiments/results/H058_one_observed_hold_FP32_qualification_root_v4/qualification_receipt.json) · [Actual wait](../../../research/reviews/H058_one_hold_ROOT_execution_source_20261016_v4/admitted_v4/worker_aux/actual_wait.json) · [Actual outer terminal](../../../research/reviews/H058_one_hold_ROOT_execution_source_20261016_v4/actual_execution_tool_terminal.root_v1.json)
