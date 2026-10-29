# Read-only performance audit, 29 October 2026

No serious efficiency failure in the inspected profile warrants delaying the
registered fits. This audit read existing code and receipts, launched no GPU
work, and changed no model, data, training or protocol sources. Further
optimization should wait for learned outcomes unless a serious failure arises.

## Useful throughput and limits

The exact run is `../runs/profile_shutdown_v2/`. Its `result.json` records epoch 2
from monotonic time 5910812.602749125 to 5910817.305431604: 33,334 canonical targets,
830 sites, 52 updates and 4.702682479 seconds, or 7,088.29 targets/s. Cold iterator
startup took 21.369801067 seconds; the second iterator took 0.003278340 seconds.
Startup materially lowers a short profile's whole-run utilization average.

Filtering `activity.json` to that second-epoch interval gives nine device-wide
utilization samples: mean 82.22%, minimum 6%, maximum 93%. These approximately
0.5-second samples describe one short warm epoch. They do not establish sustained
campaign utilization, SM occupancy, exclusive kernel activity or statistical
performance superiority. Inspect sampled compute PIDs before attributing
device-wide activity to this run.

The earlier candidate results in `../review/audit_profiles.log` report warmed
throughput of 7,361.71 targets/s for batch 16 versus 7,398.93 for batch 32. Batch 16
achieved 99.50% of the observed best throughput with 3,040,972,800 versus
5,860,414,976 peak allocated GPU bytes. Keep registered batch 16. Each candidate
had one measured warm epoch in fixed execution order; differing update counts
and online losses are not accuracy-based batch-selection evidence.

## Concrete findings

- `../root/train.py:253` synchronizes through the finite-loss `.item()`. Line 257
  calls `clip_grad_norm_(error_if_nonfinite=True)`; installed Torch 2.6 branches on
  its CUDA norm tensor, causing another host barrier. A future frozen harness
  could combine loss and norm validation into one check before the optimizer
  update, preserving fail-before-update behavior. No change is proposed now.
- Subtracting cumulative counters at updates 50 and 100 in the run's
  `progress.jsonl` gives 4.553364350 seconds wall time, 0.535673749 loader wait,
  0.181314784 transfer, 1.331890017 forward/loss checking, 2.379801559
  backward/clipping, and 0.066823040 optimizer. Host waiting and CUDA event times
  can overlap; they are not additive exclusive wall fractions. This interval
  crosses an epoch boundary. Epoch 2 alone took 0.436681065 seconds to its first
  batch, so these data do not show substantial steady mid-epoch starvation.
- `../data/site_dataset.py:140` caches immutable prepared items in persistent
  worker LRUs; hits still fingerprint both files. `joint_model_v1.py:320`
  constructs disjoint CSR blocks per minibatch, and `Batch.pin_memory` copies
  their components. These are real host costs, but measured waits do not justify
  a loader rewrite now. The 12 GiB worker cap exceeds the approximately 6 GiB
  fitting cache, avoiding capacity-driven rereads after workers encounter sites.
- Transfers run on the current CUDA stream. `non_blocking=True` alone does not
  overlap next-batch copies with current compute. The measured 0.181-second
  transfer interval limits the likely benefit from transfer overlap. Optimizer
  time is also small; optimizer fusion is not the leading issue.
- The model uses cached CSR and fused PyG `message_and_aggregate` with
  `torch.sparse.mm`: no forward routing reconstruction or graph edge-by-channel
  materialization. The width-64, three-layer structure arm performs 20 forward
  sparse aggregations including initialization/readout. SNP pooling still
  materializes SNP-row by sample by channel activations. These are useful model
  computations; do not inflate width, layers, repeated work or batch size merely
  to increase an activity percentage.

The run's `loader_shutdown.json` records four owned workers exiting with code
zero, completed iterator shutdown and no live pinning thread. No worker/pinning
leak was observed in this profile.

## Source binding

Paths are relative to `research/joint_model_20261029/`. SHA256 values were read
from completed receipts. The installed clipping implementation inspected was
`.runtime/env/lib/python3.11/site-packages/torch/nn/utils/clip_grad.py` in the
repository runtime.

| Source | SHA256 |
|---|---|
| runs/profile_shutdown_v2/result.json | dbf2aac09b7eaa1a666ba08ea5ee272cda0c5b01e3620a1e9b3601de1912df33 |
| runs/profile_shutdown_v2/activity.json | d7722714f92251a3481c74bbf0ecaf172fb48824032a84fb85ab690dad1cb20a |
| runs/profile_shutdown_v2/progress.jsonl | da6eeff6bf5667ea7dde8d68cfa6488ef157b68d3d15111d0aeab27df6a1799a |
| runs/profile_shutdown_v2/loader_shutdown.json | 8464c02aedc3b23e0b4215470f20687114a1ebd9d259d06598d708962e38e0e7 |
| review/audit_profiles.log | 0bf93ce4ff13b1d492fa791ba6978fc4f57424433db007c30eb8200541b4fd79 |
| root/train.py | 7a2d9ec4b05d38537a09cc27ae1ba262632cb2a5b9ccd1bf8b4790751c139122 |
| model/joint_model_v1.py | ba87b3b7639db4ce07b424a26d1673a043ea1d9ddbe34914d7f6397177006aa4 |
