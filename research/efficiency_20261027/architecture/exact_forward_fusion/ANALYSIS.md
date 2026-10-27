# Exact forward forcing fusion

The candidate is useful as a narrow transfer optimization. It does not reduce
the model's full node-occurrence or leaf-by-tree work. For valid finite FP32
states, selecting the original GPU upward/outside states before their CPU
copies produces the same selected values, sum and memory.U forcing. The full
CPU upward/outside/forcing/readout boundaries are retained. The custom
backward is the exact existing explicit backward function, with unchanged
saved-tensor ordering and context contract. No live executor is registered.

For G cut-group occurrences, the forward path avoids 256G bytes of selected
float transfers back to GPU and adds 8G bytes of representative indices: a
nominal 248G-byte H2D reduction. Original 256N-byte first-state D2H transfers
and CPU boundaries remain. A GPU aminmax reduction per block replaces the
CPU scan of 128N bytes of upward states. On TRAIN000, that scan reads a
26,287,266,816-byte state payload; upward/outside boundaries still occupy
52,574,533,632 bytes. These byte counts are not wall-time predictions.

First-phase GPU live temporaries grow by at most 296K + 4B named bytes,
where K is the maximum cut-group count in a block and B is the number of
blocks. At 512 holds and the conservative 1,600 groups per hold, 296K is
242,483,200 bytes. This excludes allocator and kernel workspace. Inspect
measured GPU headroom before qualification; the candidate is not admitted.

The CPU check uses three synthetic star trees, all 1,600 sample columns,
three tree blocks and three history windows in each direction, with the
actual original base, PyG sweeps, episode/cut/history computation and explicit
backward. First-state/forcing boundaries, context and every checked gradient
match exactly. All 33 exercised trunk gradients match; 12 preexisting decoder
gradients remain intact. The four site-MLP tensors are not exercised because
the input is synthetic pooled-site features. No real scientific payload,
checkpoint, optimizer update or GPU run is used. Peak RSS was about 606 MB.

The reduced finite-value guard rejects any NaN/Inf. This is stricter only for
invalid states: the previous Python max could ignore a NaN in a later block.
All valid finite-state algebra and the original denominator proof remain.

Execution owns integration. ROOT should compare complete service time with
identical thread counts and fresh optimizer/RNG state, then perform original
output/gradient/update checks before considering production use. Stage labels
now combine first and forcing work, so old individual-stage timings do not
map one-to-one. No performance or accuracy improvement is established here.
