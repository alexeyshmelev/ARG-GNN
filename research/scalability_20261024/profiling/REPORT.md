# TreeLoom isolated integration profile

Both slice runs completed with exit code 0 and passed paired FP32 parity.
The untraced warm original call took 2.0581s versus 1.2629s integrated
(1.63x in this one pair). Full-parent timing and peak memory remain unmeasured.
The frozen runner is ready for ROOT admission using the full-parent protocol below.

## What was measured

The shared input is the real TRAIN000 first 128 marginal-tree packet:
all 1600 haplotypes, 681 SNPs, physical support [0, 87959) bp and 22 actual
original fixed focal records. Both arms use the original graph/history/GRU
machinery within that window. Outside-window and future history are absent.
These results do not establish full-parent throughput, training accuracy or
campaign completion time.

Every call restores checkpoint 94 model parameters, Adam moments/counters and
saved RNG, then executes a real forward, original focal cross entropy, decoder
backward, complete trunk backward, clip and AdamW update to 95. Both arms reuse
one immutable prepared bank. All 49 optimizer counters reach 95. No production
update is saved. Original source, inputs and held-out truth are unchanged.

## Untraced measurements

Source: [slice completion](runs/slice_gpu5_v1/data/completed.json).
Common observation preparation was 0.9115s; the authentic TRAIN label join
was 0.0131s. Total diagnostic wall time was 13.3855s.

| Call | Original seconds | Integrated seconds |
|---|---:|---:|
| Same-process first call |2.5877|1.5956|
| Warm call |2.0581|1.2629|

The first original call precedes integrated; the warm order is integrated then
original. These are shared-process first-call and warm timings, not independent
process-cold measurements. A single pair on a shared device does not give a
confidence interval or isolate every cache/order effect.

| Synchronized warm stage | Original seconds | Integrated seconds |
|---|---:|---:|
| Model and optimizer transfer/setup |0.0814|0.0128|
| Layout construction/binding |0.0177|0.0010|
| Context forward |0.6150|0.3413|
| Decoder forward/backward |0.2771|0.1856|
| Trunk backward |1.0521|0.7098|
| Clip and optimizer |0.0075|0.0053|

The largest measured savings occur in context forward and trunk backward.
The decoder equations are unchanged, so its timing difference and setup
difference cannot be assigned solely to the categorical site encoder.
Nested profiler stages overlap their parents and must not be added together.

All numerical checks use atol=rtol=1e-4. Across the two integrated untraced calls,
the maximum absolute errors versus the original first call were:

| Quantity | Maximum absolute error |
|---|---:|
| Classifier logits |4.7684e-7|
| Raw pre-clip gradients |3.2783e-7|
| Updated parameters |2.9802e-8|
| Adam moments |3.9582e-9|

The original warm call also differs slightly from its first call and passes,
consistent with practical FP32 reduction variation. This is numerical parity,
not bitwise identity.

Original Torch peak allocation was 1,059,561,984 bytes at warm and integrated
was 700,943,872 bytes; reserved peaks were 1,533,018,112 and 977,272,832 bytes.
These allocator values exclude GPU context/library allocations. In-process
host RSS reached approximately 1.9 GB in the untraced run. Integrated retained
6,454,092 bytes of immutable layout within its 32 MiB cache limit: one cold build,
one warm hit and no eviction. Its receipt is captured before final cleanup,
so active_executor_leases=1 is expected; cleanup releases after backward.

## Operator trace

Source: [trace completion](runs/slice_gpu5_trace_v1/data/completed.json).
This separate warm-repeats 0 diagnostic passed parity and actual wait exit0.
Tracing increased total wall time to 53.4463s, including collection/aggregation;
its arm wall times are excluded from the timing comparison above.

| Nested device event total, ms | Original | Integrated |
|---|---:|---:|
| aten::copy_ |166.09|127.40|
| IndexBackward evaluate_function |91.51|90.61|
| aten::_index_put_impl_ |63.46|62.76|
| IndexCopyBackward evaluate_function |60.70|49.20|
| aten::linear |63.94|26.00|
| Pageable host-to-device memcpy |49.74|40.75|
| Pageable device-to-host memcpy |102.21|67.29|

The rows overlap and must not be summed or treated as disjoint utilization
fractions. Both arms show 1763 pageable host-to-device memcpy events and 145
pageable device-to-host events. Integrated still has substantial copy/indexed
backward work after reducing linear work. This slice trace does not justify a
further dense/low-rank factorization; full-parent measurements should first
establish costs and memory with the exact integrated path.

## Full-parent protocol ready for ROOT

[full_parent_protocol.json](full_parent_protocol.json) binds the same checkpoint,
original full TRAIN000 observation and all 10000 original focals. It requests
one independently prepared full parent, 32 holds per GNN block, 8 haplotypes per
full-sequence GRU tile, connected 1024-hold history windows and one warm repeat.
It runs four independent 94-to-95 updates on one shared bank and disables tracing.

The preparation limits are exactly max_nodes 400000000, max_holds 100000 and
max_episode_payload_bytes 8589934592. They match the original workflow job
call.spec.training_spec.preparation_limits, current training_v2.LIMITS and
the frozen profiler call. The current paper attempt2 job carries scientific
arguments and has no preparation_limits field.

ROOT owns admission, supervision and launch on physical GPU 5 with an 8 GiB
process VRAM target and 256 GiB host budget, allowing up to 24 h waiting and 4 h
execution. The runner has a soft Torch reserved-memory check, not a hard
allocator cap; external process-level VRAM supervision is required. Existing
healthy jobs and reservations must be preserved.

The historical exact full-parent CPU qualification covers 77472 holds,
1600 samples, 227474 SNPs and 205369272 packed node occurrences. It measured
30,005,238,092 prepared tensor bytes and 63,457,189,888 bytes preparation/pack
peak RSS, with no model/CUDA execution. One full FP32 [N,32] array is
26,287,266,816 bytes. Prepared bank plus an up/out pair and context/VJP pair
already totals114,312,302,924 bytes before autograd, history/forcing, metadata,
the 8 GiB layout cache, cold-build workspace and allocator overhead. 256 GiB is
therefore a guarded operational proposal, not a proven peak.

Smaller execution tiles retain full support and connected history; they do not
make every full-parent CPU tensor tile-local. Live compact pool contents cannot
be borrowed across processes and have no disk archive, so independent one-parent
preparation remains necessary. No full-parent success or speed claim is made.

## Frozen source and evidence

Runner: run_profile.py, SHA256
383924b779b60594c7b222439eff8a44f2fdcbb3eedc438cea8fe3093fa748e2.

Both completions record the immutable checkpoint, packet, original data and
integration source hashes. [artifact_manifest.json](artifact_manifest.json)
records delivery hashes for this report, protocol, runner and measured receipts.
No profiling-agent GPU launch or production edit was performed.
