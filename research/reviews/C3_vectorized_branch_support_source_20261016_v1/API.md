# C3 vectorized branch-support source only

This additive prototype replaces the *per-site/per-node Python descendant-count loops* inside one connected tree's branch-concordance calculation with bounded-site-batch CPU Torch int64 counts and height-level index_add_. No existing source, live encoder, job, model, guard, cap, donor-frequency routine or runtime integration was changed. The live C3 warmup remains on its current implementation.

Source file: [branch_concordance_support_torch_v2.py](branch_concordance_support_torch_v2.py).

## API

```python
support = branch_concordance_support_torch_v2(
    tree, genotype, allele_count, sample_ids,
    site_chunk_size=32,
)
```

Inputs and return match the original four-argument branch_concordance_support_v1 contract. The optional chunk size must be an integer from 1 through 256. tree is the original connected NativeObservedTreeV1-compatible observation, genotype is the original integer [S,H] categorical allele matrix with -1 for missing, allele_count is [S] with declared per-site counts1..4, and sample_ids explicitly joins genotype columns to tree.sample_id. It returns a fresh NumPy float32 [N,5] array in the original native node order, with the original field meanings and order.

The function reuses the original _validate_alleles, _integer and _layout validation. That preserves the original raw-clock/topology checks, including original raw-clock validation precision. This proposal adds no FP64 tensor arithmetic. Counts, proper-split decisions, differences, minima, tie counts and the global split count are int64/Python integers. Floating support accumulation and final normalization are FP32.

The module has no eager NumPy, Torch or PyG import. Its public function creates CPU tensors only. It never calls CUDA, changes thread settings, changes environment variables or performs file I/O. Later callers own CPU-thread limits and the original worker RSS-abort cap. Native parent/node arrays and level-wise index_add_ are directly compatible with the existing Torch/PyG observation pipeline. A MessagePassing subclass is unnecessary for this integer preprocessing operation.

No reference_class, cohort label, sample-age threshold, alpha, query ancestry truth, inferred ancestral allele or mutation placement record enters this API. All observed query and reference genotypes remain legitimate split evidence. Donor channels continue to be computed by the unmodified C3 donor_frequencies, including its original generic C-class, age-family, common-callability and alpha=.5 rules.

## Preserved equations and cases

For each original site, called samples contribute one int64 seed at their actual native sampled vertex and called allele. Internal sampled vertices retain their own seed in addition to descendant contributions. Every nonroot child sends its completed count once to its parent. Grouping by ascending child height supports unary and arbitrary-degree trees.

For each declared, observed nonempty proper allele split, retain

```text
difference = called_descendants + total_allele - 2*descendant_allele
mismatch = min(difference, ncalled - difference)
minimum = min(mismatch over ALL nonroot branches)
```

Every minimum-discrepancy branch participates, including unary ties and branches whose called descendant set is empty or whole. Both biallelic complementary splits and every proper multiallelic split are counted. Padded, absent, all-called alleles and sites with fewer than two called samples contribute zero. No branch-size or SNP-support filter is added.

Exact support divides one unit among minimum-zero winners. Closest support divides 1-minimum/ncalled among all minimum winners. Discrepancy contributes mismatch/ncalled to every nonroot branch. First accumulate all proper splits across all chunks and alleles, then divide the first three fields once by their global count. The raw split count is common to nonroot branches. Root fields0..3 remain zero and field4 is one. With no proper split, every node has fields0..3 zero and field4 one.

The C3 encode_block all-isolated-sample-forest bypass must remain as written. This public function retains the original connected-tree validation and does not synthesize a root or edge.

## Memory and work

Let N be native nodes, E=N-1 edges, H actual observed sampled vertices, S SNPs, B<=site_chunk_size<=256, A<=4 the maximum declared allele count, and L the number of occupied child heights. One dense int64 descendant tensor is 8*N*B*A bytes. The largest level's gathered source can add up to another 8*N*B*A bytes. This gives an approximately 16*N*B*A byte integer-DP payload, plus several integer [E,B]/[B,H]/[B,A] temporaries, FP32 [E,B] contributions, masks, [E,3] sums, [N,5] output and allocator/runtime overhead. It is not a measured process-RSS bound.

For default B32/A4, the main DP-plus-largest-gather estimate is 2,048*N bytes, about20.48MB at N10,000, before the listed additional temporaries and the retained existing worker RSS cap. No [S,N,A] tensor is materialized. Per-chunk temporaries die when the private helper returns. Original whole-parent observed arrays remain caller-owned.

Topology grouping costs O(N log N) once per function call, with O(N) stored schedules. Counting and split reductions cost O(S*N*A), plus O(S*H*A) seeding and O(ceil(S/B)*L) sequential level dispatches. At most four Python allele iterations run per chunk. There is no per-site or per-node Python loop in descendant propagation or mismatch scoring. Deep chains still require many small level operations, so speedup is unproved and can be poor on pathological depth. Torch startup, allocation and small-hold overhead can outweigh vectorization. The current encoder also separately computes layout, so removing that duplicated work is outside this minimal proposal.

## Precision identity and later evidence

Implementation identity: branch_concordance_torch_int64_levels_fp32_v2. Original source is src/arggnn/mutation_routed_tree_observations_v1.py:262–312, file SHA256922df286e595370ade23595420a325014e4df64aac88060e8a1a846684eab068.

The original routine accumulates in NumPy float64 and returns float32. This additive routine accumulates in FP32 with site-chunk/allele reductions. Therefore rounding can differ and may depend on chunk size. Integer split/minimum/tie decisions are preserved by design. No bitwise equality, numerical equivalence, measured speedup, actual RSS fit or scientific outcome is claimed. Routine vectorization is not a novelty contribution.

A later root-authorized small representative FP32 comparison can use an existing observed CPU-bank input/one connected hold, fixed chunk size32 and the original four input arrays. It should record original/new implementation identities, N/H/S/A/depth, native sample-ID join, elapsed wall time and actual worker RSS. Compare all five output columns and report actual max/mean absolute differences and tolerance, rather than silently overwriting original cached features. Verify split counts/root/no-evidence flags and the practical unary/missing/multiallelic/internal-sampled cases present in that representative input. A genuine mismatch in integer-derived decisions requires repair. Ordinary FP32 reduction differences are expected and accepted by the user.

This is a future bounded check proposal, not a numerical fixture suite, FP64 parity requirement, discarded optimizer campaign, root qualification or execution authorization. A future replacement must carry the new feature version and actual differences honestly. It must preserve the active C3 worker/thread/RSS controls and receive separate root integration. No current cache, runtime/source pin, resource admission, process, model, guard or cap was changed.

## Coordination and source-only verification

The maps worker supplied the exact equations and edge risks before sealing. Its design is research/reviews/C3_actual_startup_CPU_feature_diagnosis_20261016_v1/VECTORIZATION_DESIGN.md, SHA2569d1ac24f8d60ef513f41192cb605abf1ceae2694602196b5adea81455bf87f61. It explicitly confirmed global normalization, integer decisions, sampled internal seeds, all unary ties and unchanged donor/age/alpha semantics.

The only permitted checks in this task are source inspection and standard-library AST parsing. No original/vectorized scientific code, Torch import, numerical comparison, model, fit, scorer, qualification, GPU/job, installation, admission, ledger mutation or Git mutation was run. Any reported source parse result validates syntax only.
