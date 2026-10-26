# Exact preprocessing optimization report

The qualified implementation substantially reduces two measured preprocessing stages while preserving the existing scientific inputs and model. Release evidence is complete in `integration/release.json`. American training now uses the qualified four-process preparation on GPU 3, resuming from completed update 241. The first complete production input was prepared and packed in 896.095 seconds, and GPU training now overlaps preparation of the next input. The European parser and lookahead wrapper remain undeployed.

## Measured improvement

| Qualified workload | Existing | Candidate | Speedup |
|---|---:|---:|---:|
| Complete European TRAIN000 retained-site VCF conversion | 422.10 s | 46.41 s | 9.10 times |
| Complete American TRAIN000 observed preparation | 1,521.36 s | 769.01 s | 1.98 times |

The European comparison covers 1,335,446 retained SNPs and 952 haplotypes. Genotype and allele-count arrays, callbacks and source hashes match exactly. The American comparison covers the complete registered parent. All tensor bytes, shapes, dtypes and biological metadata match, excluding only runtime pointer and version stamps. Compiled serial preparation took 1,007.66 s before adding four-process execution.

These are stage measurements under shared-host contention. They are not complete-training speedups or new ancestry accuracy results.

## What changed

Integer topology traversal, exact descendant sets and cut indexing use compiled kernels. Original floating arithmetic and ordering remain intact. Independent complete trees are prepared by four spawned CPU processes with one native thread per worker, then assembled in their original order. Chunks are bounded by both tree count and bytes. An oversized tree takes the unchanged serial path without pruning. The European reader vectorizes ordinary phased calls and retains original interpretation and rejection for unusual cases.

Thread-pool encoding was tested and rejected because it was slower. Four processes were selected instead of eight because the latter provided little additional measured benefit. Final full-parent execution completed 606 chunks with no CUDA initialization or surviving worker processes. Sampled simultaneous RSS peaked at 54.79 GiB, within the qualification reserve. Admission and sampled checks are not operating-system memory limits.

## Correctness and release evidence

Tests passed 11,173 genotype cases, 1,750 VCF cases, 17 loader cases and 2,586 independent adversarial VCF cases. Structural tests passed 782 layouts, 210 episode sequences, 210 cut plans and 25 encoded banks. Independent review added 30 complete sequences, six malformed-level checks and 13 invalid kernel-boundary cases. Process tests cover 1, 2, 4 and 8 workers, chunk boundaries, byte admission, whole-tree fallback, ordering, RNG preservation and failure cleanup.

The unchanged GPU model was checked on a real 128-tree TRAIN slice with 1,600 haplotypes, 681 sites and 22 original supervised focals. Inputs are byte identical. FP32 logits, gradients, clipping, updated parameters and AdamW moments pass `atol=1e-6`, `rtol=1e-5`. Maximum candidate logit difference was 4.77e-7. An unchanged-reference repeat differed by 7.15e-7. This control supports ordinary GPU rounding variation rather than changed preprocessing. Bitwise identity of the entire training trajectory is not claimed.

Twelve authenticated proof roles, complete-parent records, continuation contracts and independent release review are linked from `integration/release.json` and summarized in `PLAN.md`. Earlier rejection-guard findings, inherited-RSS startup failure and report-serialization failure are retained with their corrections. No historical failure was relabelled as a successful run.

## Deployment boundary

The scheduler accepted the American replacement as attempt 5. It restored update 241, the original optimizer moments and RNG state. Four preparation processes are active. The task RAM reservation is 592 GiB, retaining the full 352 GiB cache and both 128 GiB working reserves. Admission retains the scheduler's 64 GiB host margin and accounts for the native Relate job, which also started. Production verification observed the first GNN sweep on GPU 3 while four workers encoded the next input. The first full input retained 76,982 trees and 1,600 haplotypes. Its loading, preprocessing and packing took 896.095 seconds. This is a different parent and a broader timing scope than the controlled TRAIN000 comparison, so no paired production speedup is inferred. The first sampled active GPU utilization was 20 percent. A completed optimizer update under the replacement is not yet claimed. Evidence is in `integration/production_verification.json`. The European parser is qualified on the complete input. The one-parent lookahead wrapper has scheduling and cleanup contract tests, including fixed-logit product parity, but no complete real prediction qualification. Both European components remain undeployed.

Frozen model architecture, simulation recipes, sample order, losses, training budgets, metrics and native competitor implementations are untouched. This release does not introduce an accuracy gain. Remaining serial work and future workloads may reveal further bottlenecks. It does not promise permanent full GPU utilization or that future optimization will never be needed.

The first migration launcher failed before signalling because this Python lacks the native pidfd API. A separately reviewed recovery uses the controller's existing Linux syscall fallback. Its tests cover unavailable APIs, changing process identity, changing harmless telemetry and prevention of duplicate dispatch. The failed attempt and original launcher are preserved. Post-migration review confirms the three pre-existing workers remain live and all 257 of their source pins are unchanged.
