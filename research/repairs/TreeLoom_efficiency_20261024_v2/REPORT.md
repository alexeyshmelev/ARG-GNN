# Read-only TreeLoom and GST efficiency diagnosis

Observed on anogena at 2026-10-24T15:54:44.287622+00:00. No live job/source was modified, signalled or relaunched. No TEST truth was opened.

## European AP TreeLoom, GPU5

The healthy legacy worker 3029584 has completed 59/90 full-parent updates, now in epoch 20. Latest loss is 1.5452; the latest retained durable checkpoint is epoch 15/update 45, with the next scheduled checkpoint at epoch 20/update 60. The last 10 update intervals average 24.1 minutes (median 23.1; range 19.3–30.3). The remaining 31 updates would take about 12.4 hours at that mean; this is conditional arithmetic.

The approximately 20.94 hours between the all-observation barrier and GPU binding was late GPU-acquisition wait. It must not be described as CPU preparation time.

Six GPU samples over 25 seconds were 34, 35, 37, 19, 12, 15% utilization (mean 25.3%), with about 53 GiB of 80 GiB occupied. Occupied VRAM is not active compute. The paired 266.8-second process sample shows 1.25 CPU cores for the worker, 56% of its CPU time in the kernel, about 51,000 minor faults/second, no new major faults and zero physical-read-byte delta. The supervisor separately used 0.51 core, almost entirely system CPU. This supports host/allocation/synchronization overhead, without proving its dominant source.

Relevant source candidates are per-update CPU layout construction, repeated CPU↔GPU bank transfers and 256-site encodings, 128-hold stable-leaf history, checkpoint backward recomputation and full-length GRU tiles of 32 haplotypes. See european_execution.py:35, 49, 54–86, 88–117, 140–151, 170–206 and european_observations.py:385–413. There is no active per-stage profile: NullProfileV1 is used by default. Exact CPU/kernel/transfer/history fractions remain unknown.

**The GST new-tree-per-window pattern does not appear in the traced TreeLoom routes.** European preparation creates one workspace and traverses ts.trees() for the whole parent (european_observations.py:317–326). American lazy-compact training delegates through runtime_v2.py:269 to static_preparation_v2.py:67–77, which does the same. American runtime speed was not measured in this diagnostic.

## GST, GPU3

The healthy four-worker attempt 2 has proved epoch 1 / 3 optimizer updates. Its epoch receipt covers all 286,594 trees, 545,101,788 real edges and 4,004,871 sites. The run is now in the next epoch; later progress may exceed the last epoch receipt. Epoch 1 training took 12.20 hours; a constant 30-epoch projection is 15.25 days, excluding startup and possible rate changes.

GPU3 was 0% in all six samples with about 1.7 GiB occupied. All four CPU workers were active at 0.87–0.93 core each, 66–68% system CPU, 227,000–273,000 minor faults/second, zero new major faults and zero physical-read-byte deltas. One was observed inside __vm_munmap. This is strong evidence of CPU input preparation/allocator pressure rather than GPU model size or disk starvation during the sample. Minor faults do not equal physical disk reads.

dataset.py:97–110 calls ts.at_index(start) anew for every eight-tree window. Installed tskit trees.py:5135–5137 constructs a new Tree before seeking; its parent-array docs at 1102–1116 cover the full global node table. These parents have 90.6–91.0 million global nodes even though marginal trees are much smaller. Repeated dense-tree allocation/reset and seeking is a strong, unbenchmarked candidate consistent with the kernel/minor-fault evidence. Encoding and tensor/FD IPC can also contribute.

A future reviewed successor should preserve a mutable tskit.Tree cursor per worker/current parent and move it from its prior position, preserving every exact eight-tree window and identity. Measure preparation time/allocation and verify input/gradient/optimizer parity before deployment. No live change was made. Do not increase worker count blindly: replicas already use approximately 202.4 GiB aggregate PSS. Final checkpoint/fit.json are intentionally written only after epoch 30 (workflow.py:146–166), so their absence does not mean zero progress.

## Implication

The large tree set magnifies engineering cost; these observations do not establish that trees should be removed or that a new scientific method is required. Additional RAM helps admission/headroom and retained banks, but does not remove repeated layout/allocation, transfer or recomputation. First profile TreeLoom's actual stages and review isolated implementation changes while preserving the healthy live runs.

Evidence: diagnostic.json, cpu_io_samples.json, gpu_device_samples.json and source_refs.json in this directory. All source references include current SHA256 hashes.
