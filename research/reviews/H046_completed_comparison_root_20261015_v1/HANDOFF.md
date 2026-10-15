# H046 comparison aggregation: prepared before evaluation completion

The source is ready. No evaluation metric JSON has been read, and no
`summary.json` or scored `REPORT.md` has been created by this worker.
Root must first notify actual evaluation completion and genuine direct launcher
wait0, or execute the command below itself after verifying that evidence.

Source:
`aggregate_v1.py`, SHA256
`a8d3ad6061afc02f32b043a3814f59f27c193b57e9379cdbfbc9c3f11a8a4dc0`.

The aggregator uses the actual H046 `selected_full_diagnostics[arm].cells`
metric JSON embedded in the genuine evaluation `completed.json`. It follows
the H045/H047 comparison layout. It never opens referenced checkpoints,
probabilities, query truth, observation arrays or target payloads.

## Root command after actual successful completion

Run from `/workspace-SR003.nfs2/ARG-GNN` on anogena:

```bash
.runtime/env/bin/python -B research/reviews/H046_completed_comparison_root_20261015_v1/aggregate_v1.py \
  --aggregate \
  --expected-eval-wait-sha256 "$(sha256sum research/reviews/H046_GPU45_flank_eval980_source_v1/admitted_evaluate/actual_wait.json | cut -d ' ' -f 1)"
```

The helper validates the actual wait hash, returncode 0, direct launcher identity,
phase/GPU/input lineage and absence of logging/supervisor errors before opening
the linked evaluation metric completion. It also requires original TRAIN
960-update success/direct wait0, all 96 prediction seals before truth, unchanged
selection and exactly eight unique metric parents per arm. Existing scored
outputs are never overwritten.

## Outputs and interpretation

Future `summary.json` includes:

- Both selected checkpoint references, 30 passes/960 updates per arm, one
  optimizer seed 101, FP32 AdamW training metadata and 31,165 parameters per arm.
- All seven cohorts and exact per-parent scoring support. Each arm has accuracy,
  NLL, macro F1, boundary F1, Brier and ECE means plus sample SD across defined
  independent parents, with explicit defined/undefined counts.
- All seven cohorts' paired effects: **foreign flank minus own state** within
  each parent, followed by the mean and sample SD of those paired differences.
  Negative NLL/Brier/ECE effects favor foreign flank.
- The original H046 utility result, recomputed from genuine metric cells and
  checked against the producer's result. It requires mean Present182 accuracy
  gain at least 0.001 and mean boundary F1 delta at least -0.01, with all eight
  Present182 boundary values defined. The boundary rule concerns the mean,
  not a worst-parent bound.

Future `REPORT.md` provides the utility conclusion, training/selection details,
a complete two-arm/seven-cohort metric table and the paired-effect table.
Accuracy/F1 are displayed as percentages, paired differences as percentage
points; NLL/Brier/ECE remain raw metric units. Machine-readable values retain
their original fractional units. Undefined boundary values are never zero-filled.
Infinite NLL is retained as an infinity marker with undefined sample SD.

SD uses ddof 1 across the independent VAL simulation parents, not training seeds.
The reused validation set supports development selection. No independent test,
original-paper reproduction, significance or novelty claim is introduced.

Preparation checked source syntax and the no-argument source-only mode. No
synthetic metric fixture, scientific experiment, GPU/guard/accounting operation,
Git operation, README update or scored-output fabrication was performed.
