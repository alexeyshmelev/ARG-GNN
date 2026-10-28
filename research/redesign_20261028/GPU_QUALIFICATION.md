# GPU qualification of the bounded prototype

All four v1/v2 comparisons passed the prospective FP32 logit, full-gradient and gradient-presence checks. All eight measured runs used only physical GPU 3, with the required process title and no other compute PID observed on that device. There were no optimizer updates.

| Arm | Focal targets per batch | Frozen v1 forward/backward | Focal v2 forward/backward | Observed ratio | V1 / V2 peak allocated memory |
|---|---:|---:|---:|---:|---:|
| no_cross | 4 | 97.21 ms | 93.79 ms | 1.036 | 0.323 / 0.082 GB |
| no_cross | 16 | 115.23 ms | 97.75 ms | 1.179 | 1.165 / 0.251 GB |
| structure | 4 | 202.35 ms | 196.93 ms | 1.027 | 0.422 / 0.325 GB |
| structure | 16 | 280.61 ms | 263.58 ms | 1.065 | 1.609 / 1.171 GB |

Each number is the median of 20 repetitions after two warmups, using the fixed final diagnostic checkpoint and the first registered fitting examples. The order was v1 followed by v2. Model weights, source observations and targets match within each comparison. These are short, fixed-batch observations. They exclude dataset streaming, optimizer work, gradient reset and numerical checks. They do not establish sustained training throughput or a statistically replicated speedup.

The largest full-gradient relative L2 difference is 1.11e-06. These checks support functional FP32 equivalence for the measured GPU inputs. They do not establish bitwise equivalence or identical optimizer trajectories. CPU tests retain five strict optimizer-parameter tolerance failures out of 33 cases, with small post-update output differences described in the independent review.

The cross-tree model at batch size 16 is approximately 6.1% faster in this observation, and its peak allocated memory falls from 1.609 to 1.171 GB. This is a useful memory reduction but a modest speed improvement. The small observation does not establish the requested sustained 80% useful device utilization. Short-window average device activity for that pair was approximately 46.6% and 40.8%. Removing work can lower device activity while also reducing elapsed time. Useful training throughput and sufficient data supply remain the relevant end-to-end measures.

The inherited depth-by-depth tree passes and Python dispatch remain plausible execution bottlenecks. The independent sliding joint-convolution line changes that layer schedule, but its complete encoder, loader and predictive value are unmeasured. It must not be ranked from a bare-layer timing.

Attempt 1 stopped before CUDA because the NVML Python wrapper was missing. Its log is preserved. Attempt 2 used pinned nvidia-ml-py 12.560.30 from a separate project-local vendor directory. No shared environment, baseline process or original input was changed.

[Independent result audit](review/GPU_PROFILE_RESULT_REVIEW.json) · [Exact profile summary](root/gpu_profile_summary.json) · [Protocol](root/gpu_profile_protocol.json) · [CPU numerical review](review/FOCAL_V2_NUMERICS.json) · [Focal executor](architecture/FOCAL_EXECUTOR_V2_REPORT.md)
