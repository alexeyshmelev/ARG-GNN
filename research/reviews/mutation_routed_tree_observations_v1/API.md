# Observed-only mutation/tree preprocessing v1

CPU NumPy implementation at `src/arggnn/mutation_routed_tree_observations_v1.py`.
No loader, target reader, fit, GPU operation or actual-parent preparation is included.
The user-authorized 1 TB limit supersedes the old AGENTS 500 GB limit. This work
performs no storage audit or bulk write.

## Qualified observations and joins

`validate_observed_sample_join_v1(original, derived, mapping,
native_sample_nodes, native_sample_times)` validates the H028/H035 observation
manifest contract, including the exact source field whitelist, source-to-native
mapping, unique actual `(sample_id, haplotype)` keys and unmodified native sampled
dates. Production requires all 280 observed haplotypes, 120 actual query, 96 pure
source proxies with 24 per ANA/CHG/WHG/EHG class, and 64 mixed reference contexts.
Only the 96 pure proxy classes survive into input arrays. The other 184 are -1.
Source cohort text and role text are consumed for validation only. Neither enters
the feature bank. A fixture override supports small handmade examples.

The return exposes `sample_id` as the original observed sample_index join key,
`native_sample_node`, `reference_class`, `sample_age`, `query_id` and
`haplotype_keys`. Node IDs and sample IDs are storage/join indices. No learned ID
embedding or ancestor identity match is produced.

`align_observed_genotype_columns_v1(genotype, source_keys, desired_keys)` requires
an exact bijection of actual sample ID/haplotype keys and explicitly reorders VCF
columns. It rejects duplicate, missing or extra keys. The existing qualified H028
`parse_observed_vcf_v1` may supply observed positions, genotypes and allele counts.
This module does not parse or open a file and does not access simulated trees.

## Native topology input

Each `NativeObservedTreeV1(parent, node_time, sample_node, sample_id)` contains
one connected native marginal tree, its complete observed sampled-ancestor union,
all 280 native sampled nodes, anonymous internals and every native unary node.
Arrays are graph-local. Parent is -1 at the root. Native endpoint ages must be
finite and have strictly positive parent-minus-child durations. Chronology is
validated in raw CPU float64 before any model conversion. The normalized
log1p(duration) channel is computed from that raw difference before it is cast
to FP32. Closely spaced raw clocks may therefore round to equal FP32 node ages
while retaining a positive edge-duration feature. The model allows those
rounded ties, rejects reversed clocks and requires positive nonroot duration
features. No epsilon, age change, pruning or GPU FP64 is introduced. Disconnected
administrative rows and unused nonsampled subtrees are rejected. The helper
allows sampled internal vertices; the accompanying model currently requires
sampled leaves, as in its qualified genealogy input.

`observed_sham_geometry_v1(TRAIN_trees)` extracts only observed nonroot durations
and sampled descendant counts. Gather these arrays from all designated TRAIN
parents, then call `fit_sham_bin_edges_v1(duration, count, split="TRAIN", bins=4)`.
This produces frozen quantile interior edges in log1p(duration) and descendant
count. A VAL fitting request is rejected. Store these exact edges before any
TRAIN target opening, and apply the identical object to TRAIN and VAL. The helper
cannot certify a caller's source/split claim, so a qualified root reader must pin
its provenance and target-before-observation sealing externally.

## Preparation entry point

```python
bank = prepare_mutation_routed_tree_observations_v1(
    trees, intervals_bp, positions_bp0, genotype, allele_count,
    sample_ids, reference_class, sample_age, query_ids, readout_x_bp,
    sham_bin_edges=frozen_train_bins,
    readout_bin_edges_bp=original_clipped_bin_edges,
)
network_bank = MutationRoutedWholeTreeInputsV1.from_arrays(bank.model_arrays())
validate_inputs_v1(network_bank)
```

Intervals are genuine contiguous marginal holds in original physical bp0.
Nothing is coalesced, resampled at scoring centers or clipped to artificial
250 kb tiles. All holds participate, including holds without sites or scoring
centers. Original VCF sites outside inferred coverage are excluded only at that
coverage boundary. Every original site inside coverage survives, including
multiallelic sites and sites not retained by genealogy inference. Missing calls
remain -1. Coverage, intervals, exact readout positions and supplied original
clipped scoring bin edges are preserved in metadata. The caller must supply
those original edges to preserve the scoring-bin receipt. Query IDs are used
only to gather final readout columns and cannot carry an input ancestry label.

`bank.model_arrays()` contains exactly the following fields for
`MutationRoutedWholeTreeInputsV1`. Metadata is excluded.

| Field | Shape / meaning |
| --- | --- |
| parent | [N] flat int64 native-parent rows, root -1 |
| node_time | [N] observed raw generations, float64 until model FP32 conversion |
| sample_index | [T,H] native rows in the persistent observed genotype-column order |
| upward_levels | tuple of int64 arrays, child-before-parent height levels |
| upward_children | each level's children of those parent rows |
| upward_parent_rank | corresponding index of each child's parent within that level |
| downward_levels | tuple of int64 arrays, parent-before-child depth levels, roots excluded |
| edge_geometry | [N,4] log1p(child age), log1p(parent age), log1p(duration), each age channel / log(10001), then sampled descendant count/H; roots all zero |
| branch_support_true | [N,5] aligned branch concordance, described below |
| branch_support_sham | [N,5] whole-vector within-tree sham placement |
| genotype | [S,H] int16 original allele lookup integers, -1 missing |
| log_donor_allele_frequency | [S,A,8] compact float32 frequencies before any softmax/centering |
| donor_call_mask | [S,8] bool common-cohort H028 callability |
| panel_allele_frequency | [S,A] observed called-panel categorical frequencies |
| site_entropy | [S] natural-log called-panel entropy |
| site_hold | [S] int64 genuine containing tree, original physical order |
| spans | [T] positive physical interval widths in bp |
| readout_hold | [R] int64 exact readout-containing marginal tree |
| query_column | [120] actual query columns for readout only |
| reference_class | [H] ANA=0, CHG=1, WHG=2, EHG=3 at pureR96; -1 at other184 |
| sample_age | [H] raw observed sampling dates |

H=280 in production. A is the maximum original declared allele count inside
coverage, at least1 and at most4. Padding allele slots are zero and never looked
up. No S*H*19 feature tensor is cached. The model gathers allele-invariant tuples
from these frequency arrays and categorical lookup indices in small chunks.

## Frequencies and masks

Eight donor groups are younger ANA..EHG, then older ANA..EHG, where observed
sample age is below350 versus at least350 generations. The group boundary does
not project or change dates. A group with no observed donors is globally absent
and remains zero/masked. At a site, if a globally present class in a cohort has
no called donor, all present classes in that cohort are masked, matching H028.
Available class probabilities are `(allele_count + 0.5) /
(called_donors + declared_site_alleles*0.5)`. Log probabilities and masks stay
separate. The model additionally masks every feature for a missing leaf call.

Panel frequency uses all called observed haplotypes with no donor label and no
pseudocount. Every called allele has positive frequency. An all-missing site has
zero panel frequencies and entropy. Neither allele index, SNP index, REF/ALT
ordering nor ancestral state receives an embedding or biological polarization.

## Branch concordance and sham

For every called nonempty proper allele split at each site, let D be the branch's
sampled descendants restricted to called samples and C the called allele clade.
Discrepancy is `min(|D symmetric_difference C|,
|D symmetric_difference complement(C)|)/N_called`. All alleles of every
multiallelic site contribute. Both complementary biallelic splits contribute.
Descendant allele counts are obtained with tree sweeps. There is no sample-pair
or node-pair matrix.

Support fields, in order, are:

1. Exact minimum-discrepancy support, divided equally among all zero-discrepancy
   tied branches, then normalized by the hold's total proper called splits.
2. Closest support, divided equally among all minimum-discrepancy ties and
   weighted by1-minimum discrepancy, then normalized by the same split total.
3. Each branch's mean discrepancy across all proper called splits.
4. Raw callable split count, common to nonroot branches of the hold.
5. Binary empty flag when no proper called split exists.

Unary branches participate independently in ties. Roots have `[0,0,0,0,1]`.
These are observed concordance summaries, neither mutation events nor branch
posteriors. All contradictions remain in the mean discrepancy.

The sham uses one fixed seed101 and no favorable reseeding. It deranges entire
five-field vectors in each `(tree, frozen_duration_bin,
frozen_descendant_count_bin)` stratum of at least4 nonroot branches. A single
randomized cycle gives no fixed assignment inside eligible strata. Smaller
strata and roots remain unchanged. Geometry and every tree's support-vector
multiset are unchanged.

Metadata reports assignment movement separately from numerical vector movement,
supported-vector movement, nonzero-field movement and individual field counts.
Supported vectors have a nonzero exact, closest or discrepancy field. Physical
weights are each branch's genuine interval span. Readout/query weights count
scoring centers in the containing hold; multiplying by the common actual-query
count leaves the fraction unchanged. Per-tree constant fields are disclosed,
including callable count and empty flags. A supported-vector numerical movement
fraction below0.5, or no supported vectors, marks the mechanism test inconclusive.
This is a diagnostic gate and never causes reseeding.

## Implementation checks and limits

The receipt records33 handmade CPU checks, including independent set-based
concordance calculation, missing/multiallelic alleles, unary ties, allele-index
invariance, sample-key and native-column permutation, pure-label/date joins,
missing/absent donor masks, genuine empty/unscored holds, original clipped bins,
sham multiset preservation and positive numerical movement. Exact model-schema
conversion and CPU topology validation passed on48 vertices,3 holds and8
haplotypes. The broad one-bin fixture is chosen before inspecting movement and
is a handmade control check, not a proposed alteration to production four-bin
TRAIN quantiles.

Preparation runtime depends linearly on site-by-native-node incidences and
alleles, with per-site N*A descendant-count temporary memory. Retained memory is
O(N + S*H + S*A*8). Only NumPy is imported by preparation. Torch/model code is
used solely in the schema integration check. Real observed arrays, query truth,
large-parent preparation, GPU execution, installs and Git actions remain outside
this assignment.
