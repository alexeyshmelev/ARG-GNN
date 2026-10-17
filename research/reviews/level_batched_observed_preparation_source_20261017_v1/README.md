# Level-batched observed branch preparation

**Source only; inactive and numerically unexecuted.** This package implements
the H060 audit section2 descendant-allele/count and branch-discrepancy primitive.
It supplies a Torch/PyG CPU kernel plus small synthetic tests for ROOT. No
production payload, model, checkpoint, truth, runtime log or live resource state
was read. No numerical test, training, inference, resource probe, install, Git,
job or lifecycle operation was performed. Source was parsed with stdlib AST
only. The parent-supplied H060 observation does not identify its phase or prove
this primitive is its current bottleneck.

## Kernel and preserved semantics

`prepare_branch_level_layout_v1` accepts the original already validated
nine-field native layout, retaining original parent/sample identities and
topological height. It does not replace strict raw chronology validation.
Parent levels are based on native topology, so sibling age ties and positive
raw branch durations that round to equal FP32 endpoint clocks do not change
routing. Real unary edges remain separate competing branches. Sampled ancestor
seeds are retained too, though H060's encoder itself requires sampled leaves.

`descendant_allele_chunks_v1` seeds every observed called sample and sums
int64 descendant counts with PyG scatter over all children of each parent
height. Its Python loops visit marker chunks and topological levels, replacing
the old site x postorder-node Python loop. There is no Python loop per site,
node or allele in the count/discrepancy hot path. All original sites and
samples remain. Marker chunks and a main-count-tensor bound limit temporary
payload. No whole-parent learned context or mutation-placement cache is added.

For each site and original allele, totals strictly between0 and called count
define the same proper splits. Both complementary biallelic splits remain.
For a native branch, the original difference is
`descendant_called + total_allele - 2*descendant_allele`; mismatch is the
minimum of that difference and `called - difference`. Minima and all ties
are computed across exactly the real nonroot branch set. The same five fields
are returned in the original order: exact split credit, closest weighted
credit, mean discrepancy, callable split count, and empty. Missing=-1,
multiallelic1..4 semantics, skipped noninformative sites and roots' empty flags
are preserved. The per-call allele width is the original maximum declared
allele count, with invalid padded slots masked, never relabelled. Donor/panel
frequencies, geometry, genotype payloads, sham construction, focal identities,
learned model, loss and target barrier are outside this kernel and unchanged.

The existing H060 all-sampled/all-isolated case stays **zero-SNP only** and
returns the same empty support without a fake root/edge. Mixed disconnected
forests and isolated holds with SNPs remain unsupported, matching the selected
H060 source. This package does not expand that scientific domain.

## Bounds and expected scope

Default marker chunk size is32 and the main int64 count tensor is limited to
8,388,608 elements (64MiB). Its shape is nodes x current_markers x original
allele_width; the chunk shrinks to fit without dropping work, and rejects if
one marker cannot fit. Count validation also runs by marker chunk. PyG scatter
outputs, gathers, mismatch masks, FP32 summaries, retained inputs, routing,
allocator state and Python objects add memory. This is not a total-RSS bound
or a measured fit. Preserve and charge existing process/cache limits.

The arithmetic remains proportional to sites x nodes x allele width. Batching
reduces Python iteration and exposes larger Torch/PyG reductions; it can add
temporary memory and dispatch/gather overhead on small/deep holds. There is no
measured speedup or demonstrated current H060 bottleneck. This addresses
observed preparation only and does not speed learned graph passes or training
updates by removing computation.

## Minimal future integration seam

Use a fresh separately reviewed encoder/source binding after ROOT's normal
decision. Keep all pinned live source and jobs unchanged. Inside each original
per-tree encoder iteration, retain the already computed native layout tuple
and use:

```python
prepared = prepare_branch_level_layout_v1(native_layout)
support = level_batched_branch_support_v1(
    prepared, g[lo:hi], k[lo:hi], original_sample_ids,
    marker_chunk_size=32, max_count_elements=8388608).numpy()
```

Keep every original layout/chronology/sample/date/leaf validation and original
tree/site interval slice around this call. Reuse that same validated layout
for geometry/levels so the old second layout call is avoided. In the actual
H060 source, all-connected blocks take the early
`_connected_encode_block` route in
`src/arggnn/american_c3_gnn_observations_v1.py`; mixed blocks use the local
`H060_isolated_hold_compatibility.../observations_v1.py` loop. A future
integration must explicitly bind the intended connected route as well as the
isolated-compatible route; changing only the local imported support name would
miss the connected fast path. No global monkeypatch is appropriate.

ROOT can initially apply the kernel only to fresh global preparation while
keeping the legacy block-seal route unchanged. Sharing per-tree observed
primitives between those stages is a separate small caller integration; this
package does not alter the provider's load/seal/release behavior or pretend
arbitrary slices of global packed arrays reproduce block bytes. Finish all
20 original parent seals and complete global preparations before external
TRAIN targets. Keep every original tree/site/sample/focal, both arms and all
scheduled updates. No new scheduler/admission framework is supplied.

## Canonical hash limitation

Counts, split membership, minima and tie sets are integer-defined. The new
summary sums/divisions are FP32; the old routine accumulates fractions in NumPy
float64 and casts to FP32. Even numerically acceptable FP32 agreement can change
canonical bytes. **Old block/global hashes are not promised and existing
hash equality guards must not be bypassed.** Existing seals/products remain
immutable. Keeping legacy block sealing retains its original bytes; a new
accelerated global preparation requires its own genuine source-bound
observation seal. If ROOT requires exact historical support bytes at a seam,
this FP32 kernel is not yet a compatible replacement there. No old checkpoint,
observation seal or scientific completion is relabelled by this package.
Numerical acceptance and actual binding/election remain ROOT's decisions.

## Tests supplied for ROOT, not run here

`test_synthetic_v1.py` contains small deterministic NumPy-reference fixtures:
binary/nonbinary/unary trees, native branch ties and complementary alleles,
permuted node/sample identities, a sampled ancestor, raw positive durations
with rounded FP32 clock ties, missing/monomorphic/multiallelic/empty sites,
zero-site isolated roots, changing chunk sizes, and failstop/error cases.
They compare int64 descendant counts exactly and the five FP32 fields to the
existing routine with rtol=atol=2e-6. There is no learned forward/backward,
FP64-equivalence campaign, production data, timing benchmark or GPU use. Raw
chronology fixtures retain the original NumPy float64 metadata policy.

After separate ROOT review/authorization, the intended small synthetic command
uses the existing qualified interpreter, for example:

```text
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /workspace-SR003.nfs2/ARG-GNN/.runtime/env/bin/python -B /workspace-SR003.nfs2/ARG-GNN/research/reviews/level_batched_observed_preparation_source_20261017_v1/test_synthetic_v1.py
```

This command has not been executed. No fixture-pass, numerical equivalence,
memory fit, production compatibility or speedup claim follows from AST parsing.
