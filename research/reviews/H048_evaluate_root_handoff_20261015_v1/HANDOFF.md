# H048 faithful evaluation: existing root route

The existing frozen H048 root driver already supports evaluation. **No new
evaluation wrapper, IO successor, guard or scientific change is needed.**
This packet contains source/metadata preparation only.

At the snapshot on 2026-10-15 at 16:41:59 UTC, H048 has **12/30 completed passes
and 384 verified updates per arm**. Its actual TRAIN completion and direct
launcher wait receipts do not yet exist. Their future hashes remain unassigned.

## Verified source and original TRAIN lineage

All **111** frozen source pins match. The source-only configuration's existing
root activation (execution_enabled, status and proposal flags only) exactly
reproduces the actual admitted TRAIN configuration, including all science and
resource settings.

| Actual immutable artifact | SHA256 |
|---|---|
| Existing root_driver.py | 49bbfa01dd8d510b8bf219d8f71e511fcb7add0d80c6da3ebe3e43a27218ad68 |
| Existing source_manifest.json | d31ac97dc1239d95b1c6b7921fefe7c38dd93b808330b84ba066804a1d173db3 |
| TRAIN launch.json | a7d709bf19c738b38582c4703cf3e413f83525c466f79f88d8052887e24497ab |
| TRAIN launch_input.json | 4a78a40adaf889244ebb332669336d3df71f9a917be82cb891a03a48446b1acf |
| TRAIN config.json | d1e619569068842f891a4f49da65dbf4ec913e36576a30b123630e8823b0ea29 |
| TRAIN runtime.json | 48e4f37ad2107fd23d0246a61624cb3e10857fcb09e70c1114676d23fa566dcd |

`prerequisite_snapshot.json` binds these refs, the existing root/common/context/
producer sources, original H044 preparation completion/direct wait0 and source
qualification receipt. The evaluation packet and GPU4/GPU5 output paths are fresh.

## Genuine prerequisites before admission and launch

The unchanged admission validates the original H048 paired TRAIN completion:
schema `H048_paired_train_completion_v1`, status
`completed_fixed30pass_paired_TRAIN`, both original arms at 960 updates,
VAL/outer truth unread, and exact original input/config/producer/preparation.

The original `admitted_train/actual_wait.json` must show actual direct launcher
wait0, no supervisor or logging error, matching completion SHA, matching launcher
PID **2911762**, TRAIN phase and physical GPU4. All six scheduled checkpoint
groups (5/10/15/20/25/30) must contain both exact original checkpoint path/SHA refs.
Admission checks metadata; the unchanged scientific reader verifies checkpoint
bytes and provenance during genuine evaluation. This worker opened none.

Root confirms that the original TRAIN lease is released before launching
evaluation. Direct launcher wait0 is not a historical descendant-drain claim.
The existing launcher must acquire the real exclusive lease, preserving title,
UUID/device eligibility and external processes. No lease or GPU probe was run
during this preparation.

## Exact root commands after genuine TRAIN completion

Run on anogena from `/workspace-SR003.nfs2/ARG-GNN`. GPU4 is the practical route
after H048 TRAIN releases it. Admission uses the existing shared wave lock so
the current980 prefix read and reserve append remain serialized with other root
accounting operations. The long scientific launch does not hold that lock.

The substitutions below compute only the genuine future receipt hashes, after
those receipts exist. No placeholder digest is supplied.

```bash
source scripts/project_env.sh

flock -x research/reviews/American_C3_short_train2mb_wave_source_v1/scheduler/admission.lock \
  .runtime/env/bin/python -B research/reviews/H048_GPU45_grid_root_source_v1/root_driver.py \
  --admit evaluate --gpu 4 \
  --train-completion experiments/results/H048_GPU4_grid_reset_train_seed101_v1/completed.json \
  --expected-train-completion-sha256 "$(sha256sum experiments/results/H048_GPU4_grid_reset_train_seed101_v1/completed.json | cut -d ' ' -f 1)" \
  --train-launch research/reviews/H048_GPU45_grid_root_source_v1/admitted_train/launch.json \
  --expected-train-launch-sha256 a7d709bf19c738b38582c4703cf3e413f83525c466f79f88d8052887e24497ab \
  --train-actual-wait research/reviews/H048_GPU45_grid_root_source_v1/admitted_train/actual_wait.json \
  --expected-train-actual-wait-sha256 "$(sha256sum research/reviews/H048_GPU45_grid_root_source_v1/admitted_train/actual_wait.json | cut -d ' ' -f 1)"
```

Review the successful admission in
`research/reviews/H048_GPU45_grid_root_source_v1/admitted_evaluate/admitted.json`
and its exact input/storage/consumer refs. Then retain a real root execution
session through terminal return for this unchanged supervisor:

```bash
.runtime/env/bin/python -B research/reviews/H048_GPU45_grid_root_source_v1/root_driver.py \
  --launch evaluate
```

Expected fresh output:
`experiments/results/H048_GPU4_grid_reset_evaluate_seed101_v1`.

Actual future evaluation launcher wait:
`research/reviews/H048_GPU45_grid_root_source_v1/admitted_evaluate/actual_wait.json`.

An already-created evaluation packet, partial packet or output is rejected.
Preserve it and inspect any failure; never replay the admission/launch.
The driver also permits `--gpu 5`, retaining the original GPU4 TRAIN lineage
and changing only derived evaluation output/job device names. GPU5 is currently
assigned to native v12, so it is an alternative only after root confirms its
exclusive lease and unchanged memory eligibility are available. Choose once.

The unchanged actual launcher requires at least **54,000,000,000 available GPU
bytes**: the original 50,000,000,000-byte CUDA cap plus 4,000,000,000-byte margin.
It preserves the eligible GPU4/5 and two-physical-device limits. Evaluation keeps
the original 100,000,000-byte output reservation, 4,000,000-byte bounded log,
512,000,000,000-byte CPU RSS cap, 32 GiB RAM prediction cap, full phase caches and
24-hour admission lifetime. Root obtains fresh actual capacity and bindings at
admission; this handoff asserts no future capacity or resource release.

## Frozen evaluation and result interpretation

Both arms retain 31,933 parameters, complete H042 PyG/SNP/BiGRU bodies, the full
original 952-input/686-query support and FP32 computation. Training remains the
same seed-101, 30-pass/960-update independent-reset AdamW condition. Evaluation
performs no optimization.

The existing producer loads both arms at all six scheduled checkpoint passes,
predicts every original position on all eight original VAL parents and seals
all **96** probability products before opening any VAL truth. It preserves the
original selection rule and full selected diagnostics for all seven cohorts,
including boundary and calibration metrics. No probability arrays are written
to disk and no outer TEST capability is admitted.

The paired direction is **graph_to_transition minus
graph_to_own_confidence_fixed_distance_chain**. The H048 utility rules differ
from H046 and must remain exact:

- Primary utility: equal-parent mean Present182 accuracy gain at least **0.10
  percentage points**, with mean 1 kb boundary F1 delta **at least zero**.
- Secondary boundary utility: mean boundary F1 gain at least **1.00 percentage
  point**, with mean accuracy delta **at least -0.05 percentage points**.
- Both boundary rules require all eight primary paired boundary values defined.
  Report paired sample SD across parents, one optimizer seed. Any H045 global-best
  comparison is separate. Reused VAL supports development, not independent
  confirmation, operator priority or a biological-rate interpretation.

After genuine evaluation, root verifies producer completion
`H048_paired_VAL_completion_v1` /
`completed_original8VAL_paired_development_selection`, all-96-seals evidence,
actual launcher wait0 and absence of operational errors before reporting scores.
No H048 accuracy or utility outcome is asserted here.

No existing file, live source, guard, journal or checkpoint was edited. No
scientific payload, query truth, probability or checkpoint was opened. This
worker performed no actual admission, execution, guard/accounting operation,
GPU operation or Git operation. No synthetic campaign or new review layer was
introduced.
