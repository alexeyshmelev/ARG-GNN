# Bounded GPU stable-leaf history cold qualification

This source-only successor keeps H050 stable_leaf_memory, FP32, native parents,
1600 haplotypes, all original 10000 focals and complete 30Mb support.
It has no active authority. ROOT alone supplies the exact finite admission
request, reviewed manifest, genuine controller binding and storage transaction.

## Corrected learned-history execution

The original per-hold global index_select can create a complete [K,8] gradient
for each native hold. The new history implementation gathers once from a disjoint
per-window forcing slice, uses the same sequential recurrence, saves CPU boundary
states and recomputes at most 128 holds for each CUDA VJP. It returns the incoming
state VJP across every window, so there is no history truncation. The CPU
coefficient equations, CPU group pooling and subtract-forcing residual remain.
Only the redundant whole-history checkpoint is removed; local recomputation is
inside the custom autograd Function. No cumulative-product/reassociated scan.

For 128 holds, 1600 haplotypes and rank8, one float32 leaf/state window is
6,553,600 bytes, local forcing is at most that size and int64 indices are
1,638,400 bytes. Full forcing, output, gradients and boundary checkpoints remain
on CPU. These are tensor-size bounds, not measured GPU/RSS peak or speed claims.
The new 24GiB allocator and 128GiB owned RSS limits are explicit ROOT downward
failstops from the original 64GiB/384GiB limits. Whole-work limits reject;
they never authorize cropping. Scientific update order and fixed epoch30 remain.

## Qualification route

qualification_ROOT_builder_v1.py --prepare-ref <exact active ROOTrequestref>
authenticates the new independent source pass and immutable original evidence,
performs the existing actualv3 owner4 accounting transaction with <=64 direct
consumer bindings, writes truthful source-binding runtime metadata and returns
the new parent argv. Its default CLI remains inert.

The worker first executes one tiny original-versus-windowed FP32 history
forward/forcing-gradient/ell-gradient comparison on guarded GPU0. It then loads
the genuine complete observation, calls the immutable vectorized optimized
preparation directly with no cache producer/metadata prerequisite, seals its
whole-parent plan, joins only authorized TRAIN000 targets and performs one
disposable full-parent optimizer update. Genuine direct/outer waits and owned
drain remain required before any acceptance. No all49 synthetic optimizer
campaign is run. The old scientific config and all49 actual gradient/AdamW
checks remain unchanged for the full-parent update.

A five-second stdlib sampler starts before Torch imports. It atomically replaces
one primary-scope profile.latest.json (<=48KiB), retaining <=24 stack signatures
of eight frames, <=48 stage timers and the last16 entry/exit events. It has one
bounded transient profile.next.json, no trace archive or per-sample file.
Output directory enumeration/checkpoint writers share the profiler lock.
The main thread's Python stack, source function and active stage locate long
CPU phases even while GPU0 is idle.

## Evidence and limits

One tiny CPU FP32 fixture was actually run:9 holds,1600haplotypes,45groups,
rank8,three-hold windows, unequal spans and coefficient underflow. Forward and
forcing VJP errors were exactly0; ell_theta VJP max error9.5367431640625e-7,
within rtol3e-5/atol2e-6. There were no optimizer updates, actual scientific data
reads or GPU operations. GPU scatter FP32 accumulation can differ in final bits;
bitwise GPU parity is not claimed. The guarded cold run must establish actual
fit, useful progress and the full30Mb optimizer result.

The parent, borrowed GPU attachment, NVML inventory and target resolver are
copied from the current frozen reviewed source, with only c namespace imports
changed. The source-delta check confirms eight tree/forcing/GRU executor method
bodies are unchanged. No old/live attempt was edited, signalled or relaunched.
