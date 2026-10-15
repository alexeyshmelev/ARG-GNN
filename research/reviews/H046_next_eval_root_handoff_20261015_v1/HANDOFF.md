# H046 next evaluation root handoff

Prepared route only. Root owns admission, GPU scheduling and execution. This
packet adds no wrapper, guard or scientific change. The executable entry point
is the existing frozen `H046_GPU45_flank_eval980_source_v1/root_driver.py`.

## Current source and metadata evidence

See `prerequisite_snapshot.json` for the timestamped metadata snapshot and hashes.
All 87 evaluation source pins match. All 81 original source pins are unchanged.
The evaluation scientific configuration is byte-identical to the actual TRAIN
configuration, SHA
`13d1b2fa0503a452c3bee7ba3b47bf3d702ade18a796c0f4c6e562e6f129f5a8`.
The actual original TRAIN launch/input/runtime pins and the immutable H044
preparation completion/direct-wait0 pins match. No checkpoint, observation,
target, query-truth or other scientific payload was opened.

At the preparation snapshot TRAIN has not supplied both terminal receipts.
A pass receipt is a completed-update lower bound, not terminal completion.
The admission packet and GPU4/GPU5 evaluation outputs are fresh. Do not run the
commands until the actual TRAIN prerequisites below hold.

## Exact prerequisites enforced by the existing evaluation source

- Original TRAIN `completed.json`: schema `H046_paired_train_completion_v1`,
  status `completed_fixed30pass_paired_TRAIN`, arms `foreign_flank/own_state`,
  960 updates each, VAL/outer truth unread, and original input/config/preparation.
- Original `actual_wait.json`: actual direct launcher wait, returncode 0,
  no logging or supervisor error, matching completion SHA, launcher PID
  2811715, phase train and physical GPU5.
- Six original checkpoint groups 5/10/15/20/25/30, each with both exact original
  checkpoint path/SHA references. Admission checks this metadata. The unchanged
  scientific reader verifies checkpoint bytes only at evaluation execution.
- Root confirms original TRAIN lease release. The original launcher closes its
  lease in `finally` after its direct worker wait. The existing evaluation
  launcher must acquire the actual exclusive lease before CUDA. Stale JSON text
  in `.runtime/locks/gpu_5.lock` is not evidence of a held or released lock.
  No descendant drain or historical writer-absence claim is inferred.
- The existing admission revalidates the frozen source closure, genuine H044
  completion/direct-wait0/drain, count-derived caps, current prospective980
  capacity and direct bindings. Root reviews returned admission metadata before
  invoking launch. Never replay an existing admission or output.

## Scheduling

Use GPU5 evaluation first, then the queued native v12 GPU5 attempt. Retain
GPU4 for the active H048 job. The existing evaluation driver also allows
`--gpu 4`; this changes only the device and derived fresh output/job names,
while preserving the original GPU5 TRAIN lineage and scientific config.

The frozen actual launcher uses an exclusive per-device project lease. GPU4
therefore cannot overlap H048 through this launcher while H048 holds that lease.
Either eligible device also needs at least 54,000,000,000 currently available
bytes: the unchanged 50,000,000,000-byte CUDA cap plus 4,000,000,000-byte margin.
The launcher checks the actual UUID, eligibility, process title and two-device
limit. Do not bypass it or change the existing policy to rearrange this queue.

## Root commands after actual TRAIN terminal evidence and lease release

Run on anogena from `/workspace-SR003.nfs2/ARG-GNN`. These are direct calls
to the existing driver, with no new runtime wrapper. The two SHA substitutions
bind the genuine receipts that do not yet exist at preparation time.

```bash
cd /workspace-SR003.nfs2/ARG-GNN

.runtime/env/bin/python -B research/reviews/H046_GPU45_flank_eval980_source_v1/root_driver.py \
  --admit evaluate \
  --gpu 5 \
  --train-completion experiments/results/H046_GPU5_flank_train_seed101_v2/completed.json \
  --expected-train-completion-sha256 "$(sha256sum experiments/results/H046_GPU5_flank_train_seed101_v2/completed.json | cut -d ' ' -f 1)" \
  --train-launch research/reviews/H046_GPU45_flank_root_source_v2/admitted_train/launch.json \
  --expected-train-launch-sha256 a899af3d485339b26e1873eb13d7db0b514c4886b668a5e81127c29e8d20cd0e \
  --train-actual-wait research/reviews/H046_GPU45_flank_root_source_v2/admitted_train/actual_wait.json \
  --expected-train-actual-wait-sha256 "$(sha256sum research/reviews/H046_GPU45_flank_root_source_v2/admitted_train/actual_wait.json | cut -d ' ' -f 1)"
```

Review the successful admission in
`research/reviews/H046_GPU45_flank_eval980_source_v1/admitted_evaluate/admitted.json`
and its input/storage references, then run the supervisor to its genuine terminal
return in a root execution session:

```bash
.runtime/env/bin/python -B research/reviews/H046_GPU45_flank_eval980_source_v1/root_driver.py \
  --launch evaluate
```

Expected fresh output:
`experiments/results/H046_GPU5_flank_evaluate_seed101_eval980_v1`.
Actual evaluation launcher wait:
`research/reviews/H046_GPU45_flank_eval980_source_v1/admitted_evaluate/actual_wait.json`.
Root verifies evaluation completion and actual successful wait plus released
lease before starting native v12 on GPU5. Checkpoints and failed artifacts remain
preserved. Root owns any completion accounting and subsequent score publication.

## Unchanged scientific interpretation

The same 8 VAL simulations evaluate both trained arms at the six original
checkpoint passes, using the fixed original query support, grid, metrics and
selection rule. All 96 prediction products are sealed before VAL truth opens.
This is development selection evidence with one optimizer seed, not an
independent outer-test result. Caps remain 100,000,000 output bytes,
512,000,000,000 CPU RSS bytes, 50,000,000,000 CUDA reserved bytes, 4,000,000
log bytes and a 24-hour admission lifetime.

No admission, launch, GPU operation, guard mutation, scientific execution or
Git operation was performed in preparing this handoff.
