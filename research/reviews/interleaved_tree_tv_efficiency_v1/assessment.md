# H033 narrow efficiency assessment

This read-only source assessment identifies serial Python dispatch around small tree solves as a concrete batching opportunity. It does not establish a measured CPU-time attribution or speedup.

## Existing evidence

The first actual TRAIN tile, bio_paths_pilot_train_000:1, contained 79 packed blocks, a maximum 21224 vertices per block and 1635385 total replicated vertices. Its recorded timings were:

| Arm | Forward seconds | Backward including clipping seconds | Peak CUDA reserved bytes |
| --- | ---: | ---: | ---: |
| interleaved | 2.8903 | 5.3412 | 513802240 |
| blocked | 2.5396 | 5.1679 | 513802240 |

The geometry receipt reports at most 684 vertices per observed augmented tree across the prepared tiles, with approximately 1.63–1.68 million replicated vertices per tile.

interleaved_tree_tv_gnn_v1.py:359–365 serially visits every block for each of three A calls: 237 custom solves per forward on this tile. _eliminate_v1 at 249–260 launches indexing/arithmetic kernels through Python upward/downward level loops. Nonreentrant checkpointing adds 237 recomputed forward eliminations and 237 adjoint eliminations during backward, giving 711 elimination passes per complete training forward/backward.

InterleavedTreeTVBankV1.to at 81–85 and PackedTreeBlockV1.to at 63–68 also issue many individual tensor transfers. The driver recreates device banks at 442 and 514 for TRAIN and VAL tiles. These transfers precede the measured forward timer. Their wall-time share is unmeasured.

The supplied instantaneous observation of 0% GPU utilization and about 1.1 CPU cores is consistent with dispatch or transfer overhead, but a snapshot alone does not establish the cause.

## Minimal batching proposal and unmeasured bounds

Increase ragged packing from 32 to 256 readouts in _pack_readout_forests_v1 at 207, with matching constructor/config admission validation at module 331 and driver 78. Preserve every complete per-readout forest, reference origin, actual Q row, source placement, solver, TV update, output order, optimizer step and selection rule.

For a 2500-readout tile, block count falls from 79 to 10. This reduces block-driven solve invocations from 237 to 30 per forward, and forward/recompute/adjoint elimination passes from 711 to 90 per training pass. It also reduces per-block transfer and checkpoint bookkeeping. These are source-derived counts, not measured speedup estimates; level depth and kernel work remain.

Using the observed maximum 684 vertices/tree, a 256-readout block has at most 175104 vertices. One four-channel FP32 buffer uses at most 2801664 bytes (2.80 MB), and one scalar FP32 precision buffer uses 700416 bytes (0.70 MB). Total replicated vertices and persistent geometry remain unchanged. Additional simultaneous transient buffers should cost tens of MB, but allocator behavior, saved state and the actual peak require qualification; the 514 MB measured peak is not an upper bound for the proposed packing.

Use a separately frozen implementation revision, verify tiny forward/gradient equivalence within the existing FP32 tolerances, and measure its first real update before adopting it. Do not mutate the running frozen H033 job. No new profiling, experiments, tensor reads or source changes were performed for this assessment.

## Evidence bindings

- src/arggnn/interleaved_tree_tv_gnn_v1.py: SHA256 c173280d1206bb89e99997079234920bc93bc51b2db0a6b674d0b9c173552d48
- src/arggnn/train_interleaved_tree_tv_gnn_v1.py: SHA256 ba7964aeec1327f9a3adee8eeb362d29af09230e0794f6a64b8f3421f92c788b
- experiments/results/interleaved_tree_tv_gnn_v1_seed101_GPU2/first_training_tile.json: SHA256 7bd231aef15fb76aeefaf6511898b33127c3923ffa51de8fb3a1a0611fa0fb81
- experiments/results/interleaved_tree_tv_gnn_v1_seed101_GPU2/geometry.json: SHA256 f2906b1ab8042b9d1203a0a14789346044b265681f2236a145af14ed79c06560
