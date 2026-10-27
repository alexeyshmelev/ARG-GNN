# TreeLoom efficiency and accuracy work

The goal remains unmet. Target sustained useful GPU utilization of at least 80 percent, with accurate ancestry inference. Measure complete preparation and training, useful targets per second, memory and storage. Device activity alone is not an accuracy or efficiency result.

Use physical GPUs 3, 4 and 5. Keep all project computation and writes under the repository on anogena. Preserve competitor algorithms, the original simulation data, scoring and historical results. The project limit remains 1 TB.

## Current work

| Ticket | Status | Next action |
|---|---|---|
| E01: full-parent exact executor | Deployed | American attempt 7 continues from checkpoint 268. Checkpoint 276 is durable. |
| E02: sustained utilization | Open | Observe the next production executor across cold preparation and warm updates, with compute-client identities recorded. No 80 percent claim yet. |
| E04: boundary fusion and GPU residency | Qualified | Combine the accepted implementation with preparation changes in one checkpoint transition. |
| E06: concurrent parent preparation | Report repair | Serial compute finished, but the oversized completion report failed before the concurrent arm. Use bounded signatures, then rerun both arms. Preserve encoded values, parent order and RNG. |
| E07: combined deployment | Source v3 qualified | Scheduler admission now requires 76 GiB free GPU memory. Finish the repaired preparation comparison, review terminal evidence and preserve the latest durable checkpoint for one transition. Not deployed. |
| E08: memory lifetime and GPU context | CPU tests and source review passed | Release upward states after their last use, then assess keeping final context and its gradients on GPU. Separate candidate, no production changes. |
| E09: monitoring overhead | Measured, not changed | One live process PSS scan took 1.95 seconds. Record actual callback frequency and cost in the successor before changing scheduling. |
| A01: European observed cohort inputs | Resource scheduling repair | First qualification was stopped by the supervisor before numerical testing. Restore the eight observed cohort channels supplied to the CNN after stable admission. Fixed 90-update fit is prepared separately. |
| A02: independent development validation | Implementation | Prepare a separate 10 Mb VAL with unchanged demography, roster, rates and native inference. Keep original 200 Mb TRAIN and TEST unchanged. |
| B01: ancestry benchmarks | Running | Finish American TreeLoom, native ARGformer, GST, ARGMix and direct four-class comparisons. Report only completed common scoring. |

## Accepted measurements

Each full comparison below is one fixed-order pair on a shared GPU. Do not multiply these gains or present them as measured whole-training acceleration.

| Change | Reference update | Candidate update | Result |
|---|---:|---:|---|
| CPU8, explicit backward, 512-tree blocks | 620.745 s | 380.415 s | 38.7 percent lower time. Allocated GPU memory 51.73 to 7.91 GB. Deployed. |
| Fuse consecutive tree operations | 324.486 s | 298.470 s | 8.0 percent lower time. Qualified. |
| Keep upward states and direct gradients on GPU | 318.315 s | 237.006 s | 25.5 percent lower time. Allocated peak 59.85 GB, reserved peak 64.80 GB. Qualified. |

All full qualification pairs preserve all 77,472 trees, 1,600 samples and 10,000 registered targets. Logits, loss, all 49 gradients, 49 updated parameters and 147 Adam tensors passed the declared FP32 tolerances. No predicted class changed. These are numerical execution comparisons, not new ancestry accuracy results.

The first production observation across cache preparation and two resumed updates averaged 18.35 percent device utilization. The resident candidate's full call averaged 69.09 percent on a shared GPU. Neither establishes exclusive TreeLoom utilization or the requested sustained 80 percent.

Current European TreeLoom accuracy is 54.80 ± 1.02 percent across the same five test chromosomes, versus 92.70 ± 0.35 for the historical AncestralPaths CNN. TreeLoom is not leading. The input audit found that its current adapter discards observed cohort information supplied to the CNN. The new condition addresses that mismatch. Architecture development will use TRAIN and independent VAL, not iterative TEST tuning.

## Closed investigations

| Candidate or issue | Conclusion |
|---|---|
| Level-local sweep variants | Slower in repeated GPU comparisons. Do not repeat without a new bottleneck hypothesis. |
| Context-only residency in the earlier executor | Full call was slower than the selected CPU-context executor. Not deployed. |
| Fixed-input GPU transfer pack | Only about 1.72 percent projected full-parent saving. Closed without a production restart. |
| TreeLoom-Transport | Full update took 1,122.875 seconds, with no validated accuracy advantage. No larger learning pilot. |
| Profiler v1 attribution | Invalid: CUDA annotations were counted as kernels. Corrected physical-event accounting reports 36.76 percent kernel activity and 64.56 percent activity including copies for the fused prefix. |
| Adaptive preparation v1 accounting | Phase and child allowances were not jointly admitted. Corrected and reviewed in v2. |
| Initial successor callback identity | Failed before restoration. Corrected using the actual module execution route and tested before attempt 7. |

## Scientific and operational decisions

- Exact execution changes preserve outputs, gradients and optimizer behavior within declared FP32 tolerances. Changed architectures get separate names and checkpoints.
- The cohort8 fixed final-epoch fit can proceed without checkpoint selection on VAL. Independent VAL is required for architecture selection and comparative development claims.
- The 10 Mb VAL is a separate development condition, with prospectively fixed seeds and coordinates. It does not reproduce the papers' 200 Mb tables or establish full-length behavior.
- The long-form paper and substantive repository audits are recorded by the agents. Missing archival metadata and incomplete graphical extraction remain explicit. Source coverage is not proof of reproduction completeness.
- Use one combined production transition. Prepare and review its source first, confirm terminal comparison results, preserve the latest checkpoint, wait for actual owned-process exit, remeasure resources, and submit through the existing queue.
- Never count hypothetical freed memory as actual launch capacity. Never change unrelated processes.

## Evidence

- [Full residency review](fidelity/UPWARD_RESIDENCY_FULL_V2_COMPLETION_REVIEW.json)
- [Full boundary fusion review](fidelity/BOUNDARY_FULL_V1_COMPLETION_REVIEW.json)
- [Corrected profiler review](fidelity/BOUNDARY_ACTIVITY_V3_COMPLETION_REVIEW.json)
- [Concurrent preparation source review](fidelity/TWO_PARENT_PREPARATION_V1_SOURCE_REVIEW.json)
- [European cohort qualification wrapper review](fidelity/EUROPEAN_QUALIFICATION_CLI_V1_SOURCE_REVIEW.json)
- [Short validation feasibility](fidelity/EUROPEAN_SHORT_DEVELOPMENT_VAL_FEASIBILITY_v1.json)
- [Monitoring cost](execution/MONITOR_COST_READONLY_v1.json)

Earlier findings, rejected implementations and exact measurements remain in the research directories and Git history. Update this current plan rather than appending contradictory status snapshots.
