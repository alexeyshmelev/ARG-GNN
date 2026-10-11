# H030 source handoff

The scientific design is frozen. This packet contains source readiness, handmade CPU checks and an inactive storage template. Root owns independent review, registration, active admission, reservation, lease and GPU qualification.

## Final source pins

| Path | SHA256 |
|---|---|
| `src/arggnn/reference_logcavity_gnn_v1.py` | `8970aa2e82a43b45d1ec10b2eaae6e2203dab12356a4b968bea3c65b2e21fc0b` |
| `src/arggnn/train_reference_logcavity_gnn_v1.py` | `06c42c7c930c5415e2e6a12763d1bd42410c3dfaea100222641f0a5e492a64e4` |
| `configs/reference_logcavity_gnn_v1.json` | `828c519c13fe1b9e31da2c8b8482050779b5931645a8877c50ae9b0df72ef794` |
| `docs/reference_logcavity_gnn_v1.md` | `55f525240d5e8c36397ba0207ab6ab0f16dbac271053272d780f9999f777ca46` |

`execution_guard_ready=true` means exact V7 adapter imports/pin and direct-write guards are complete. Active input and actual reservation checks still gate execution. Do not flip that flag after this freeze. The adapter is `src/arggnn/project_budget_v7_consumer.py`, SHA256 `496cbaef484b2ff3cfc583a5e9320661d1d2d2518ef9bd21d50218f96e78aafa`. Final integration changed no science. The final config-only correction labels resource boundaries V7 rather than obsolete V6.

## Commands

From `/workspace-SR003.nfs2/ARG-GNN`:

```sh
source scripts/project_env.sh
.runtime/env/bin/python -B -m arggnn.train_reference_logcavity_gnn_v1
```

The default command reports source hashes only. Author ran it without `--execute`. Root's later command, inside the reviewed held lease and active admission:

```sh
.runtime/env/bin/python -B -m arggnn.train_reference_logcavity_gnn_v1 \
  --execute --input "$H030_INPUT_PATH" \
  --expected-input-sha256 "$H030_INPUT_SHA256"
```

## Required active input

Use schema `reference_logcavity_gnn_input_v1`, `root_authorizer=root_agent`, `fit_authorized=true`, `status=active`, `seed=101`, `device=cuda`, `epochs=10`, a canonical `job_id`, a fresh repository-relative `output_directory`, valid expiry/active-authority fields required by the unchanged input consumer, and root-approved `max_cpu_rss_bytes` / `max_cuda_reserved_bytes`. Pin both H030 model and driver in `source_sha256`. Logical job must match held lease.

Every `refs` entry is exact `{path, sha256}`. Required names:

- `config`: final H030 config above, with helper hashes intact.
- `runtime`: root's exact reviewed runtime record.
- `storage`: root's separate active V7 record constructed from inactive `storage_template_2GB.json`.
- `emitter_bindings` and `target_bindings`: original four-parent joins and C4 targets/support.
- `frozen_baseline_source`, `frozen_baseline_config`, `frozen_baseline_completed`, `frozen_baseline_checkpoint`: exact immutable selected H025 genomic smoothing epoch 10, seed 101.
- `strong_reference_completed`, `strong_reference_config`, `strong_reference_checkpoint`: exact completed selected H026 `native_reference`. Reuse checkpoint, sealed predictions and metrics without refit. Its input retains exactly the same observed parents, emitter binding and H025 checkpoint.

`observed_parents` retains all four original TRAIN/VAL parents; each entry supplies `prepared_manifest`, `observed_parent_entry` and `observed_roster`. Preserve observed manifests, tsinfer producer, role/date/reference-label joins, all 280 observed native samples and all 120 query columns. Geometry is sealed before opening TRAIN target arrays.

Root supplies final V7 core/launcher pins in storage only. The exact reservation binds those components plus adapter, H030 driver/config and runtime. Initial legacy/supplemental/journal pins must match. Output parent exists, output is fresh, and reservation is 2,000,000,000 bytes. The author template remains inactive and cannot execute.

## Numerical and scientific scope

Pure R96 observed proxy labels provide evidence. Younger sources are valid analytic 350-generation points; older sources are actual inferred sampling nodes. Positive age-potential edge costs telescope under serial subdivision and are zero below 350 because a common reference-independent factor cancels in normalized fields. Unit family/class fields normalize relative logs with explicit empty masks and no floors. Older ANA remains exactly zero.

Handmade CPU checks passed all-arm zero-head frozen-skip parity, a hand-derived field oracle, finite empty/underflow VJPs, exact and learned source-subtree exclusion, finite nonzero learned gradients, and exact zero residual with no reference support. Kernel serial-subdivision parity passes. Nonlinear GNN subdivision changes, as expected. The tiny sham has singleton bins and zero changed-lookups; root must inspect meaningful changed-lookup fraction on actual geometry. No actual payload, optimizer update or GPU qualification was performed by the author.

Five arms use one copied seed-101 initialization: learned `logcavity_only`, `static`, `carrier_gru`, `carrier_matrix`, `sham_ancestor`. Each fits 10 epochs and 160 updates over 16 complete TRAIN tiles; 16 VAL tiles and full Q120 roster remain. Frozen H025 skip, loss, support and scorer stay unchanged. Select earliest minimum equal-parent VAL Present64 NLL. First actual TRAIN-tile synchronized forward/backward timing, finite gradients and CUDA peak are recorded per arm; those updates count toward 160.

Reference tree kernels, sum-product/rerooting, TGN and DeltaNet are established. This is not a Felsenstein likelihood or ancestry posterior. Priority of the census coupling remains unresolved. Contextual cavity differences do not identify causal recombination effects. Conservative writes remain attachment-key changes, direct source relocation or episode creation, not every remote cavity change.

## Resources

Existing target-free metadata has 431–618 holds/tile, 15,956 across 32 tiles, max 276,800 native vertices/tile and conservative max 449,840 augmented rows. New CPU bank adds about 56 bytes/row, max 25.2 MB/tile. One log8 tensor costs 32N bytes, max about 14.4 MB. These exclude saved autograd and allocator overhead; H030 GPU peak and seconds are unmeasured.

Bidirectional carrier FP32 state is 128 bytes/live carrier. Recurrent arms use 862–1,236 directional boundary batches/tile. Planned total is 800 optimizer updates. Template has 4,791 parameters; nominal active counts: kernel-only 36, static 3,076, GRU 4,759, matrix/sham 3,607. Static active blocks contain 3,204 scalars including 128 zero-memory head weights. Four older-ANA kernel-head weights are data-unidentified.

2 GB covers probability-only current/best seals and last/best checkpoints. Conservative raw probability peak is 844.8 MB; H026 comparator files are reused. Prune superseded nonbest seals before next VAL. Native-tree temporary writes have exact hashed payload+8,192-byte allowance, before/after/cleanup guards. `tmp` creation has 8,192-byte allowance. Existing CUDA qualification has 32 MB before/after allowance. Actual feasibility remains for root's first tile.

## Discriminating evidence

Kernel-only versus frozen H026 tests learned age kernel plus census-source policy; H026 instead retains younger donor-age weighting. Static versus kernel-only tests learned whole-tree propagation. Matrix versus matched GRU and internal-history sham tests structured updating/coherent history under common conditional read. Require practical accuracy/NLL gains consistent across parents while preserving class recalls, calibration, short tracts and boundary precision/F1. Boundary spam or class tradeoffs do not establish biology. One seed and reused development parents support discovery only. Predictive gains do not establish priority.
