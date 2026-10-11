# H029 implementation handoff

Source is implemented and frozen for root qualification. No actual-data prepare,
GPU forward, fit, install, experiment registration, reservation mutation or Git
action was performed by the author. The original independent critique/science
packets remain unchanged. Root owns the H029 registration and all execution.

CLI source report:
`python -B -m arggnn.train_edge_innovation_gnn_v1`

Leased execution:
`python -B -m arggnn.train_edge_innovation_gnn_v1 --execute --input ROOT_INPUT_PATH --expected-input-sha256 ROOT_INPUT_SHA256`

The active admission schema is `edge_innovation_gnn_input_v1`, retaining H027's
BudgetV4-compatible input shape. Require root_authorizer=root_agent,
fit_authorized=true, status=active, device=cuda, seed101, epochs10, job_id,
creation/expiry times, output_directory, CPU RSS/CUDA reserved caps, original
observed_parents, and source_sha256 binding both new Python files. The refs are
config, runtime, storage, emitter_bindings, target_bindings,
frozen_baseline_source/config/completed/checkpoint. Reuse the exact selected H025
genomic_smoothing epoch10 checkpoint, original four parents and frozen Q120
columns. Storage must bind this driver/config/runtime, the logical job and a
separate reservation of at most1GB. No V5 migration is required.

Shared weights are copied bit-for-bit from one seed101 model template. GRU and
matrix/sham each have16 carrier state scalars per direction. All templates have
4755 scalars. Effective active counts: Static3072, GRU4755, matrix/sham3603.
Static has3200 scalars in active module blocks, of which128 head weights multiply
the identically zero memory streams. GRU has1152 additional active update weights,
so there is equal state capacity, not exact parameter parity.

All neural sweeps run on complete native forests, retaining every original
vertex/edge and adding analytic350 cut points. Younger R proxies move to those
points, older proxies remain at native sampling nodes. Actual query sample IDs
gather their350 points. Frame joins are qualified adjacent child episodes.
Absence/unqualification/reappearance and tile ends reset. Events are attachment
or direct source relocation/creation, not all remote cavity changes. GRU and
matrix receive identical cavity payloads, retention/write gates, resets, gaps
and current-key bilinear whole-tree read. The GRU comparison tests update rules;
it does not establish novelty of conditional memory or TGN.

The decoder has the same linear value maps and reference gates in Static and
memory arms. Static zeros the history streams. Its capacity includes a second
complete-tree pass. Same-age full-roster centering and the zero bias-free head
keep baseline logits exclusively in the frozen skip.

The target-free geometry seal is saved before TRAIN target arrays open. Training
uses the original exact-span loss and unchanged scorer. The driver retains all
class/boundary/short-tract/calibration diagnostics and chooses the earliest
minimum equal-parent VAL Present64 NLL. Previous nonbest probability seals are
pruned before new seals, preserving last/best checkpoints and current/best seals.
Full historical BudgetV4 checks run at preparation/epoch/output boundaries.
Per-arm checks only validate the GPU lease, logical job, RSS and CUDA cap.

Only hand-derived CPU trees were run. Checks cover initial exact skip, zero
reference/zero residual, finite nonzero encoder/update gradients through every
level-wise index_copy_, subtree exclusion, outside innovation of an unlabeled
query carrier, an ancestral edge affecting another query before pooling,
internal sham lookup changes with fixed sample histories, native older nodes,
all native vertices, absence/reappearance resets and bidirectional tile endpoints.

Existing target-free inspection JSON reports431–618 holds per tile,15956 holds
over32 tiles. Each recurrent tile consequently needs862–1236 boundary batches.
Live state is128 bytes per carrier across both directions. For N augmented rows,
single encoder/decoder hidden tensors are64N/192N bytes respectively, before
saved autograd states. Actual CUDA peaks and seconds per tile are unknown.
Root's first-tile forward/backward qualification must determine the wall/GPU
budget. Avoid a duplicate whole-preparation job. There are160 updates per arm,
640 total, plus full VAL passes across16 tiles for10 epochs.

The narrow census-specific coupling and its empirical use remain hypotheses.
TGN, GraphSSM, tree cavities, matrix associative recall and gated delta algebra
are established. Fixed physical decay, date/inference errors, sparse references,
sham singleton bins and repeated development selection remain test assumptions.
A matrix gain over Static alone cannot establish useful ancestral history.

The runner records first_training_tile.json once. It synchronizes CUDA only for
that first TRAIN tile's forward/backward timing, records finite loss/gradient
norms, allocator peak, hold/vertex counts and active parameters per arm, and
preserves these actual first optimizer steps in each arm's160-update total.
No separate data preparation or qualification run is required.
