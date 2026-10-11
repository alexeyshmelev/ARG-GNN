# Dedicated observed-parent loader v1

Source: `src/arggnn/mutation_routed_parent_loader_v1.py`.
No teacher, emitter, target binding, true genealogy, census flag, ancestry path
or population-label payload is read by this module.

`load_parent_observations_v1(parent_id, refs, reader, scratch_directory=None)` accepts one of the four
frozen development parents and the H035 `observed_parents[parent_id]` dictionary
with exactly prepared_manifest, observed_parent_entry and observed_roster.
Supply `fit.ConsumerReaderV1()`. The wrapper verifies project-local paths and
exact SHA256 bytes. Traversal, external, truth, oracle, census, target-binding
and population-label payloads are denied before calling the inherited reader.

Return: `LoadedMutationRoutedParentV1` with native tree holds, original physical
intervals, original VCF allele observations, actual sampled IDs, only96 pure
proxy labels, observed raw dates, actualQ120 readout IDs, original scoring
centers and clipped bins, a full scoring parent and an observation receipt.
This function was not invoked on actual arrays during implementation checks.

Source pins derive exclusively from observation metadata. Native TS comes from
entry.inferred_tree with the exact documented heterochronous tsinfer/tsdate
producer and prepared-manifest hash. Source roster comes from
entry.original_observed_sample_manifest. Original genotypes.vcf,
site_coordinate_manifest.json and scoring_coordinates.npz come from the
original observed directory and entry.input_observed_sha256. Prepared/entry
source, native roster, tree and fixed-grid hash joins must agree.

All280 actual sampled haplotypes, unique actual sample ID/haplotype keys,
original phased diploid order and exact native sampled dates are checked.
PureR96 has24 per ANA/CHG/WHG/EHG class. Other184 inputs are unknown. The source
manifest spells class0's cohort Ana; the frozen array helper spells it ANA.
Original manifests stay intact. Only that known spelling is canonicalized in
temporary join-check copies, without changing classes, roles, dates or IDs.

VCF parsing uses the qualified H028 parse_observed_vcf_v1. Every original
multiallelic site and missing call survives. Tree mutations and inferred
ancestral alleles never generate the genotype matrix.

`extract_native_tree_holds_v1(ts, join, coverage, offset=coverage_left)` retains
every genuine marginal hold and the complete union of paths from all native
sampled nodes. Disconnected administrative rows are excluded. Native unary
nodes survive. No analytic cuts,250kb tiles or coalescing are introduced.
Only native parent relations and raw node times enter extraction. Native
float64 dates and positive raw durations are checked on CPU. The receipt counts
rounded FP32 parent-child clock ties without changing dates or topology.
The root topology probe reports one such positive-duration rounded tie in VAL000.

Verified TS bytes are decoded through an auto-deleted temporary file in the
optional keyword-only scratch_directory. The root trainer must pass
output.path / "observation_tmp" so each arm stays inside its exclusive output
allowance. Supplied directories must be project-local, contain no traversal or
symlink components, and are created only during an authorized tree load.
Standalone calls without a directory retain the project-local fallback
.runtime/mutation_routed_parent_loader_v1/tmp. Temporary files are always deleted
at context exit. The original tree path is never reopened without verification.
All scientific arrays and preparation order are unchanged.

`prepare_parents_v1(parent_refs, reader, scratch_directory=None)` loads all four observations, gathers
observed duration/count geometry from both TRAIN parents, fits four-quantile
sham bins exactly once, then prepares TRAIN and VAL banks with unchanged edges.
It never receives target bindings. The driver must seal the receipt before
opening TRAIN targets.

Return bundle:

```python
{
    "parents": {
        parent_id: {
            "model_inputs": MutationRoutedWholeTreeInputsV1,  # CPU FP32 bank
            "observations_receipt": dict,
            "scoring": dict,
        },
    },
    "train_parent_ids": list,
    "val_parent_ids": list,
    "sham_bin_edges": dict,  # JSON-safe globally frozen TRAIN bins
    "receipt": dict,
}
```

Scoring includes all inherited fit.scoring_parent_v1 fields: parent_id, split,
samples, query_slots, raw_query_nodes, cohorts, positions, edges,
retained_indexes, coverage_original and query_ids. Query order is original
observed roster order. Target-mass and scorer helpers therefore receive the
same120 original query identities and original100bp bins. Aliases are
positions_original, bin_edges_original, query_sample_index and
original_query_nodes, plus prepared/entry references. Query cohorts are scorer
metadata and never become model arrays.

Genuine native coverage and scoring support differ at some tails. VAL000's
coverage starts273 while original scoring bins start300. VAL001's coverage
ends1999724 while bins end1999700. Every native hold remains in recurrence,
including unscored portions, and scoring edges remain unchanged. The frozen
preprocessor's optional edge argument requires full genealogy coverage, so this
loader leaves it unset and preserves original edges in scoring and the receipt.

Receipt:20 focused checks including four-parent metadata/roster integration
through fit.ConsumerReaderV1, hash boundaries and denied payloads, a handmade
two-hold native TS with unary/administrative nodes, sampled-column permutation,
raw-clock rounding and unchanged scoring-grid tails. Only16 actual metadata
and roster files were read. No actual VCF, scoring-array or native-TS array was
opened, no actual bank prepared, and no GPU, fit, install, scheduler, storage
audit or Git action occurred.

The narrow worker-scratch correction adds four path-only checks for a supplied
project path, external path, traversal and a symlink component. No actual array
load or scientific rerun was performed. The receipt now records24 checks.
