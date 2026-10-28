# Bounded observed TRAIN windows, 28 October 2026

This isolated data prototype preserves the frozen European TRAIN registry rows,
target labels, inferred trees, sample identities and observed genotypes. It does
not train a model, read TEST files, create simulations, or change author baselines.
Its finite genomic context defines a new TreeLoom condition. It does not claim
numerical equivalence to the previous whole-parent model.

## Actual bounded extraction

`train000_first_window.npz` and its JSON receipt contain the four earliest
position-sorted TRAIN000 records, with original example indices 6667–6670. They
are all BAA examples at the same observed SNP. This deliberately convenient
engineering fixture is unsuitable for learning, class-coverage or accuracy claims.
It contains one focal tree and two complete neighboring trees on each side:

- 5 trees, 9,515 packed nodes and 9,510 child-to-parent edges.
- All 952 observed haplotypes in each tree: 686 queries and 266 pure references.
- Every observed SNP inside the five intervals: 39 sites by 952 haplotypes.
- Four unchanged labels, joined by example_index rather than position-sorted row.
- Local coverage [429997, 437062.5), with core [434087.5, 436438.5).
- Original coordinates equal local coordinates plus 397 bp for this parent.

The initial extraction took 0.250088 seconds, with process peak RSS 826,564 KiB,
2,973,268 logical VCF bytes read and a 709,850-byte uncompressed NPZ output. The
run used a filesystem session already accessed by preceding metadata/index
inspection. Cold-cache status was not controlled, page-fault I/O was not measured,
and this is not a complete-TRAIN speedup or throughput measurement. The source
8,791,314,172-byte tree file was mapped read-only rather than loaded into a full
resident tskit TreeSequence. Inference metadata is 293,476,778 bytes because it
embeds the complete site mapping; the loader reads a 16-KiB prefix for needed
references and the coordinate offset, not that whole site map.

The initial packet predates the optional coordinate-array extension. Its exact
schema is recorded in `train000_first_window.json`. Current source exports extra
coordinate and context-proposal arrays documented below. Preserve the initial
packet as the measured fixture; use a fresh output name for subsequent probes.

## Exact observed-tree extraction

`KastoreArrays` reads the Kastore v1 header/descriptors and memory-maps typed
column arrays. It checks format, bounded descriptor count, offsets, lengths,
unique keys and current file size/mtime. It executes no serialized program.

`ReplacementTreeReader.at(x)` uses binary search in the native insertion and
removal order indexes. Let I be the number of edges with left <= x, R the number
with right <= x, and G the newest equal-left insertion group. The fast path
requires every edge in G to contain x, all edges in G to share one right bound,
and |G| = I - R. Since I-R is exactly the number of active edges, this proves G
contains every active edge at x. To establish the complete native interval too,
the next insertion must not occur before the shared right endpoint, and the
latest preceding removal must not occur after the insertion-group left endpoint.
Otherwise the fast path rejects the source for generic handling. The result
retains every sample, including isolated
samples. It does not assume a binary tree, one root, consecutive internal IDs,
or a fixed number of nodes. Local parent times must strictly exceed child times.
Internal IDs serve provenance only. They are never used as biological identity
between trees. Only the observed sample index and (sample_id, haplotype) persist.

This is an exact fast path for the actual complete replacement Relate intervals,
not a general random-access tskit algorithm. A persistent older edge causes an
explicit error. `generic_small_reader` provides a tested generic tskit fallback
for already bounded sources (default maximum 32 MiB). It refuses the 8.8-GB source.
There is no silent simplification or automatic full-parent fallback.

Consecutive windows follow left/right interval boundaries with half-open [l,r)
semantics. A target exactly on a breakpoint belongs to the right tree. Genomic
coverage borders clip the available number of halo trees and set explicit
boundary flags. Missing labels and exceeded node/site bounds are errors, never
reasons to discard an example. Structural error resolution must retain the
original label draw through a larger approved limit or explicit microbatch split.

Small registry, manifest and target-array checksums are verified. Large source
checksums remain the inherited authenticated references from the earlier audit;
this probe does not claim to rehash the tree, full VCF or huge inference receipt.
Read-only stat snapshots detect file size/mtime changes after a handle opens.
This is not a substitute for production admission checking immutable source
bindings once outside the per-sample hot path.

## Observed features and truth boundary

All nine observed cohorts are separate categories, including present_day. Leaf
features are 9 cohort one-hot columns, 5 reference columns (Ana, CHG, WHG, EHG,
unknown), log1p(age_generations)/log(10001), and query/reference role one-hots.
The separate raw reference_class vector is -1 for every query. Cohort identity
is observed metadata, not locus ancestry. Query ages, query genotypes and all
sampled leaves are retained. Target labels are returned only in the `labels`
array, never passed to graph, cohort or genotype construction. The model adapter
must whitelist its input tensors and consume labels separately at the loss.

`SortedVCF` opens the original, sorted, single-chromosome plaintext observed VCF.
It checks header sample columns against the full paired haplotype manifest,
binary-seeks file byte offsets, and requires an exact integer original position
before converting to VCF's one-based convention. Local predecessor/successor
checks reject duplicate POS and local non-increasing positions; whole-file
sortedness is inherited from the frozen observed-input contract, not rescanned.
Calls must be phased diploid nucleotide alleles. Multiallelic calls up to four
alleles and missing (-1) calls are preserved. Genotypes come from the observed
VCF, not mutation reconstructions or ancestry truth. All observed sites inside
each retained complete tree are returned, including trees selected by sparse
coordinate sampling. A selected full tree may extend beyond the requested
coordinate domain; those extra observed sites are retained and disclosed.

## Target sampling, duplicates and variable batch size

The empirical objective remains L(theta) = sum_i ell_i(theta) / N over the exact
N=50,000 registered TRAIN example identities. Duplicate biological site/query
pairs, if present as distinct registered examples, retain their separate IDs and
multiplicity. IID sampling with replacement also retains repeated draws of one ID.

`uniform_target_draws` samples a global integer in [0,N), then maps to parent and
local example_index. Thus P(parent p)=n_p/N and P(i|p)=1/n_p, giving P(i)=1/N.
The unweighted mean of B IID selected losses is unbiased for this objective at
fixed parameters. With random context, ell_i denotes the expectation under the
declared context sampler, not the old whole-parent loss function. Parents are not selected uniformly when their row counts differ.
A complete random permutation is also suitable for exhaustive epoch exposure;
it must keep every registered example rather than selecting all targets in a
convenient window. If an alternative IID proposal pi_i is introduced later, use
ell_i/(N*pi_i), averaged over the original B draws, rather than a normalized
importance-weight ratio. No class-dependent proposal is implemented here.

`batch` deduplicates identical context windows only for computation. Only the
selected focal arrays contribute loss. No additional labels from core/halo trees
are added. `tree_core_mask` identifies the containing focal tree, not eligibility
for every other label inside that tree. Different context specifications or
focal coordinates in the same native tree keep distinct window keys.

`node_budget_microbatches` returns complete-window partitions and target draw
indices. Every selected draw appears exactly once; one over-budget window fails
rather than losing its target. Sum per-target losses divided by the original
fixed B across these microbatches, then make one optimizer step. Taking separate
steps from differently sized grouped-window means can alter the stochastic
objective and is not the sampler implemented here. For a full uniform batch
covering multiple parents, retain the global B denominator across parent packets
as well; do not replace it with each parent's returned local packet length.

## Optional genomic coordinate context

`uniform_coordinate_context(rng, focal, radius, K, sequence_length)` returns the
forced focal coordinate plus K-1 IID uniform coordinates in the clipped domain
[max(0,focal-radius), min(sequence_length,focal+radius)). All complete trees
containing those coordinates are retained and ordered. The core tree is always
present, even if the random draws all land elsewhere.

`batch(indices, context_specs={example_index: spec})` preserves each coordinate
occurrence and its containing-tree mapping even when several draws hit the same
tree. Random draws have continuous density q=1/domain_width and quadrature weight
1/(K-1). The forced focal has continuous density 0, weight 0, and an explicit
forced mask. It remains available for the focal classifier and is excluded from
the random continuum estimator. All numerical metadata is finite.

These random weights estimate a normalized uniform spatial expectation. Multiply
by the domain width to estimate an unnormalized integral. A normalized
kernel-weighted context is a ratio estimator, not an exactly unbiased normalized
kernel mean. Uniform genomic sampling already selects longer intervals more
often, so multiplying by interval width again would double-weight them. The
contiguous-window mode instead provides deterministic midpoint coordinates with
interval-span-normalized quadrature weights and context kind 0. IID coordinate
context has kind 1. This extension implements no new optimizer or evaluation.

## Packet interface and parallel preparation

Core graph arrays are parent[N] (-1 roots), edge_index[2,E] (child to parent),
node_time[N], node_original_id[N], node_sample_index[N] (-1 for internal),
node_tree_index[N], and leaf_node_index[T,S]. Leaf metadata is supplied both as
leaf_features[S,17] and separate reference_class, sample_age, observed_cohort
(integer index into COHORTS), and query_mask arrays.

Tree arrays give local and original intervals, window index, core mask and
coverage-edge masks. Site arrays give tree index, native site ID, both position
conventions, phased genotypes[J,S] and allele count. Focal arrays give selected
example IDs, tree index, sample index, both positions, native site ID, genotype
calls[F,S], allele strings/counts, and separate labels[F]. Current coordinate
context arrays are context_coordinate_local_bp, context_tree_index,
context_window_index, context_quadrature_weight,
context_proposal_density_per_bp, context_forced_focal,
window_context_domain_local_bp and window_context_kind. Preserve masks and
proposal weights when deduplicating graph computation.

Dataset handles are created lazily and keyed by process ID. Pickling clears
mapped/tree/VCF handles so spawned CPU workers reopen their own read-only handles;
a forked child also reopens rather than sharing a seekable VCF cursor. Multiple
workers can share the kernel's read-only file pages. CPU worker count and the
number of simultaneously active parent handles must be selected from measured
RSS/I/O, not extrapolated from this one-window run. No multiprocessing performance
or full-parent cache build was performed. Variable-node packets should be
prepared by CPU workers and collated as disconnected PyG forests; their compact
sample mapping supplies stable leaves across the ordered trees.

The fixed-seed plan-only script `plan_uniform_batch.py` can describe a 32-draw
TRAIN000/001 batch without opening either scientific tree or VCF. Exporting that
batch is deliberately left to a separately requested run. `test_window_data.py`
uses hand-authored tskit tables and VCF fixtures, not simulation, plus the saved
real TRAIN packet. It checks generic tskit equivalence, unary/multiroot and
coverage boundaries, persistent-edge rejection and bounded fallback, exact
coordinate joins, missing/phased/multiallelic calls, duplicate POS, source mutation,
sampling multiplicity and distinct coordinate contexts.

## Independent review correction

Independent review demonstrated that active-count equality alone authenticates
only the active graph at x, not its entire interval. A partial future insertion
at 5 with current edges spanning [0,10) incorrectly returned [0,10) at x=2;
the native interval is [0,5). A symmetric previous-removal event can make the
reported left endpoint too early. The reviewed correction adds both neighboring
event guards above and rejects these unsupported cases without changing the
source tree. Both hand-authored fixtures are now regression checks, and the
independent initial failure log is retained under ../review/. The original
real-packet extraction and timing remain preserved, not relabeled after this
source correction. A revised one-window equivalence receipt checks all original
packet fields against current extraction; it is separate from that initial
measurement. No ancestry result or training conclusion follows from the repair.

## Versioned real coordinate probe and prospective plans

After independent review passed both interval-boundary regression fixtures,
`extract_coordinate_probe_v2.py` re-extracted the initial contiguous packet and
matched all 31 original arrays exactly. The separate receipt is
`interval_revision_equivalence_v2.json`; the original packet and timing were not
overwritten.

The subsequent `train000_uniform_coordinate_window_v2.npz` contains uniformly
chosen TRAIN000 example 2754, a present_day query, using seed 20261028. It has one
forced focal coordinate and four IID coordinates within a 50,000-bp radius. The
five containing trees have 9,515 packed nodes and 100 retained observed sites.
All 952 observed haplotypes, metadata and target identity are preserved. The
coordinate-to-tree mapping, native intervals, original-coordinate offset, forced
mask, continuous proposal densities and quadrature weights are recorded in the
packet and its matching JSON. Forced q/weight are zero and each random draw has
q=1e-5 per bp and quadrature weight 0.25.

This coordinate extraction took 0.154527 seconds and read 8,017,051 logical VCF
bytes, producing 770,870 bytes. Peak process RSS was 1,389,068 KiB, including the
preceding real equivalence probe in the same process. Cache state remains
uncontrolled. The single target is not a learning or accuracy diagnostic.

The plan-only uniform64_fit000_001 and uniform128_fit000_001 JSON files describe
IID registry draws from TRAIN000/001, with TRAIN002 named as a prospective
internal development parent. They create no windows or feature caches and alter
no original split. Their sampling objective uses the 33,334-row two-parent
support, not the full 50,000-row benchmark training support. A future internal
pilot must disclose that distinction; comparison to author models trained on all
50,000 rows requires the root's separately declared final-refit protocol.
