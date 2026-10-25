# GST-LAI cursor and recovery repair

The failed 24-hour fit retained one complete epoch and **three durable optimizer
updates**. Any later unrecorded updates are unknown. No checkpoint exists in its
output: the original workflow saved weights only after epoch 30 / update 90.
Its trained state is lost, so the first repaired fit must start from the original
seed. The failed output and actual wait evidence remain untouched.

The isolated successor is ready for ROOT review and scheduling. No GPU fit,
prediction, queue repair or request submission was performed.

## Cause and measured repair

The existing dataset creates a new tree with ts.at_index(start) for every
consecutive eight-tree set. tskit explicitly documents this as linear traversal.
Four persistent preparation workers were already enabled, but each worker still
repeated tree construction and interior seeking on every requested window.
Observed TRAIN000 has 95,625 marginal trees and 90,940,327 unique nodes.

The replacement retains one live tree cursor per worker's active parent and
advances through the skipped trees to its next assigned window. It encodes only
the exact requested eight-tree set. Parent changes and reverse/random requests
retain correct behavior; pickling clears cursor/workspace ownership.
Persistent workers, parent shuffle, window size/order, complete final sets,
native encoder, focal identities, labels, no-focal forwards and dropout calls
are retained. The released author GraphSetConv and LAI model are imported unchanged.

A bounded CPU probe authenticated and loaded the existing observed TRAIN000
tree file, then compared the same eight interior windows / 64 complete trees:

| Measurement | Seconds |
|---|---:|
| Common tree-sequence load |29.749|
| Original initial interior seek |3.683|
| Seven subsequent fresh original seeks |27.022|
| Cursor initial interior seek |3.689|
| Cursor advances to the same seven later starts |0.0456|

All active native node IDs, parents and intervals matched exactly. Peak RSS
was 14,529,544,192 bytes, below the probe's 32 GiB address-space cap; wall and
CPU limits were 180 s and 120 s. It opened no VCF, ancestry labels, truth or
model. These timings isolate topology access and establish no end-to-end
production training speed or GPU throughput.

## Small CPU qualification

[cpu_smoke_v3/receipt.json](cpu_smoke_v3/receipt.json) passed in 25.9 seconds
with CUDA disabled and synthetic inputs only:

- 60 strided windows across two shuffled parent passes and six random/repeated
  accesses matched every original Window/bank tensor exactly, including final
  partial, zero-SNP, no-focal and unary-branch cases.
- Two actual four-worker loader passes emitted the exact original order and
  tensors. Worker PIDs persisted across passes; all four exited with code 0.
- Original and cursor logits, raw gradients, parameters and Adam state matched
  exactly with native GST dropout enabled.
- A saved/reloaded parent-boundary checkpoint resumed the next update exactly
  across all 80 learned tensors and Adam counters. Python, NumPy and Torch RNG
  matched after continuation. The test includes rolling back the extra main
  RNG draw from a newly constructed DataLoader iterator.
- Atomic latest replacement, invalid binding/counter rejection and the
  16 MiB write cap passed. The actual synthetic recovery file was 625,710 bytes.

The first two qualification attempts failed during source-reference construction
because the interpreter path was a symlink. Canonical path handling was fixed;
their directories/log evidence is retained. The successful v3 receipt pins the
final execution sources. There was no failed scientific run or GPU allocation.

## Recovery and unchanged budget

The workflow writes recovery_latest.pt after each complete parent optimizer
step. It stores CPU parameters, all optimizer state/counters, RNG, exact epoch
and completed-parent prefix, full visit counts/history and original input/source
bindings. A temporary file is capped at 16 MiB, flushed and atomically replaced;
latest plus staging has a 32 MiB peak allowance. Only latest is retained.

A future resume reauthenticates original observations, reconstructs the unchanged
epoch shuffle and starts at the next unfinished parent. It restores RNG after
the fresh loader iterator's extra base-seed draw. An interrupted partial parent
is recomputed from the last completed boundary; no partial gradients or update
are credited. Resume requires the exact same isolated source and input bindings.
It does not import the old failed epoch receipt as model state.

Training still requires all three original TRAIN parents, all native
trees/edges/sites/samples, all frozen focal labels, 30 epochs and exactly 90
AdamW updates with the original settings and fixed final epoch selection.
Final checkpoint and fit schemas remain compatible with the original consumer.
No production GPU resume or full real-parent optimization has been tested.

## Inert scheduling handoff

[ROOT_request.json](ROOT_request.json) contains proposed fit/predict factories
and the unchanged score task, with source pins and observed failed/blocked
attempts. It is explicitly a preparation artifact, not a campaign submission.
ROOT must refresh the current revision, preserve/archive the failed output,
prepare the scheduler's required archive evidence and admit resources before
submission. It preserves the existing 24-hour wall limit, 320 GiB fit and
512 GiB prediction RAM reservations and original output caps.

GST must wait while its reservation would crowd the full profile and other
healthy jobs. [diagnosis.json](diagnosis.json) records failure/checkpoint
evidence; [artifact_manifest.json](artifact_manifest.json) binds this delivery.
