# Observed-only biological Relate chain v1

This draft infers four development parents from their exported observations. It
never opens the parent simulation manifest, source trees, cleaned oracle trees,
ancestry targets, or census targets. The primary condition is the Relate tree
sequence after five population-size iterations; the separately preserved
pre-size condition uses initial haploid N=30,000.

Relate 1.2.4 research use was already authorized by the user on 2026-10-10.
The installed official static archive and executable/helper hashes are verified
before execution. Required notice: **Copyright © 2018, University of Oxford**.
No new install, redistribution, or licence-consent action is performed.

## Inputs and adapters

Parents are `bio_paths_pilot_train_000`, `bio_paths_pilot_train_001`,
`bio_paths_pilot_val_000`, and `bio_paths_pilot_val_001`. For each parent the
only input directory is
`data/biological_paths_execution_pilot_v1/<parent_id>/observed/`.
The whitelist is `genotypes.vcf`, `sample_manifest.json`, `sample_order.txt`,
`sample_ages.generations`, `population_labels.txt`,
`site_coordinate_manifest.json`, `uniform_recombination_map.txt`,
`scoring_coordinates.npz`, and `observation_manifest.json`. Original bytes are
hashed before and after the native chain and must be unchanged.

The observed panel contains 140 diploids / 280 haplotypes: 96 pure-source
reference haplotypes, 64 mixed reference-only haplotypes, and 120 queries.
Only pure references have observed source-class labels. Query and mixed-reference
classes must be null. Cohorts and dates come from the observed roster; no local
ancestry label is constructed for any query.

The script keeps phased, segregating biallelic nucleotide sites with MAC >= 1.
Multiallelic/nonnucleotide and monomorphic exclusions are counted. Missing,
unphased, reordered, or invalid genotypes cause failure. Original VCF REF/ALT
polarization is preserved; no ancestral FASTA or unobserved polarization is
invented. VCF positions are joined exactly to the observed site manifest.
The official `ConvertFromVcf` output is checked for identical retained alleles,
phased haplotype columns, and site order. Any further converter attrition is
reported. The original converter `.sample` is preserved; the separate canonical
`.sample` duplicates the observed name into ID1 and ID2 when the converter uses
NA for one ID. This adapter never reorders diploids. The map adapter changes
only the header to `pos COMBINED_rate Genetic_Map`; the original numeric rows
remain `1 1 0` and `2000001 1 2`. Synthetic contig 22 and a uniform map do not
assert GRCh37 coordinates or an empirical recombination map.

## Native chain

Each parent is processed sequentially in a fresh output directory:

1. `ConvertFromVcf` and `GenerateSNPAnnotations` from the observed cohorts.
2. `RelateParallel.sh`: mutation rate 1.25e-8, initial **haploid** N=30,000,
   four threads, memory argument 2 GB, explicit seed, observed sample ages in
   generations, map, annotations, and checked haps/sample.
3. `ConvertToTreeSequence` for the pre-size condition. Its `.anc`, `.mut`, and
   native `.trees` are hashed and verified unchanged after the size chain.
4. `EstimatePopulationSize.sh`: exactly five iterations, one thread,
   mutation rate 1.25e-8, explicit seed, observed poplabels,
   `--threshold 0 --years_per_gen 28 --noplot`.
5. `ConvertToTreeSequence` for the post-five-iteration primary condition.

Threshold zero is an explicit observed-only protocol choice that retains
intervals; it differs from the helper's implicit threshold 0.5. The helper reads
sample ages from the initial `.anc`; it has no sample-ages command-line option.
Years-per-generation controls `.coal` reporting, while sample ages remain in
generations. `--norm_mutrate` and `--noanc` are omitted, so mutation-rate
normalization is disabled and final branch-length re-estimation runs. `--noplot`
is a standalone flag and removes the R plotting dependency.

Seeds in the config use a new SHA256 domain:
`ARG-GNN/Relate/biological_paths/five_iter/v1/{parent}/{initial,size_base}`,
first eight bytes big-endian modulo `(2**31-100)` plus one. They were checked
against the eight simulation seeds and each other, not every historic inference
seed. The size helper uses base+1 through base+5; its final branch-length step
intentionally reuses base+5.

All native/helper argv, command hashes, outer return codes, elapsed times, and
log hashes are recorded in `commands.json`. Helpers run under `bash -x` to retain
the internal trace. Initial-helper background child return codes are not
separately claimed. A failed command stops the parent, preserves logs and partial
outputs, and writes failure status; there is no retry or silent replacement.

## Actual sample IDs and coordinates

The VCF header, converter `.sample`, and every retained phased `.haps` row
establish diploid and haplotype-column ordering. The pinned converter source
appends sample nodes in that haplotype order. The script enumerates actual
`NODE_IS_SAMPLE` node rows from each native inferred output and compares the ages
at those node IDs to the ordered observed ages. It does not assume IDs 0..279,
and it does not inherit cleaned-oracle sample IDs. The source commit is ordering
and coordinate implementation evidence, not a signed identity assertion for the
release binary. Clipping preserves all inferred node IDs and sample flags;
node count and mapping are checked afterward.

The native converter uses VCF positions. The script subtracts one and clips to
`[first retained SNP bp0, last retained SNP bp0 + 1)`. Cleaned trees use local
coordinates within that span, with the exact inverse:

`physical_original = local_tree_position + coverage_left`

The original parent interval is always physical `[0, 2000000)` and its
normalization length stays 2,000,000 bp. Uncovered tails are excluded without
extrapolation. Grid index i is physical center `50 + 100*i` for i=0..19999;
retained indexes are those centers inside coverage. To query a cleaned inferred
tree at original center x, use local position `x - local_to_original_offset_bp`.
Each native site joins to
its original observed site ID, VCF position, original bp0, and local coordinate.
Common pre/post support and fixed-grid intersection are also recorded.

## Output manifest and consumer contract

The final path is
`data/biological_paths_execution_pilot_v1/relate_inferred_v1/manifest.json`.
It must have status
`four_development_parents_inferred_pre_and_post_five_iterations` before consumers
use any graph. During a partial run it is `running`; failure is
`failed_partial_preserved`. Parent records are in `replicates` and include:

- `parent_id`, `split`, `status`, and `truth_or_oracle_read=false`.
- `inferred_tree`: post-size cleaned tree path, SHA256, and condition.
- `pre_size_inferred_tree`: separately hashed pre-size cleaned tree.
- `observed_sample_manifest`: post-size derived roster path and hash, plus
  original observed roster path/hash. The file is a JSON list of all 280 observed
  rows. It retains role, cohort, age, sample name/haplotype, and reference class;
  `inferred_tree_node_id` and its compatibility alias `observed_node_id` identify
  actual post-size samples. `source_observed_node_id` is provenance only.
- `sample_index_to_inferred_tree_node_id`: all 280 explicit mapping objects.
- `pre_size` and `post_five_iterations`: complete condition-specific tree hashes,
  mappings, separately hashed rosters, coverage, site joins, grid indexes, and
  converter-order provenance. Pre-size IDs must be read from its own mapping.
- `coordinate_contract`: original interval/normalization, post-size coverage,
  local offset/transform, hashed original scoring grid, retained center indexes,
  and the location of the site join.
- `native_input_provenance`, `commands_provenance`, and
  `completion_provenance`: exact input-order evidence, command journal path/hash,
  and completed parent `inference.json` path/hash. The final manifest adds the
  latter hash after writing the receipt, avoiding a self-referential file hash.

Raw pre/post converter trees are preserved separately from cleaned observed
outputs. Cleaning drops metadata, populations, individuals, migrations,
provenance, and non-sample flags; it retains inferred topology, times, observed
sites, and mutations. No true ancestry metadata is imported.

## Root review and execution

The script is draft-only until root reviews/freezes its bytes and the config.
This producer agent did not parse real genotype records or execute inference.
A narrow handmade schema check covers VCF+1, multiallelic filtering, ID2=NA
normalization, and noncontiguous actual sample IDs; it runs no native commands.

Root may validate observed manifests without genotype-record parsing:

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python scripts/infer_biological_paths_relate_v1.py
```

After verifying source completion, input joins, resource/storage admission, and
frozen hashes, root runs from `/workspace-SR003.nfs2/ARG-GNN` in its guarded CPU
wrapper:

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python scripts/infer_biological_paths_relate_v1.py \
  --execute --expected-config-sha256 <root-reviewed-config-sha256>
```

The target must not exist. Any failed or partial run is preserved for review;
future attempts require a separately reviewed output/config rather than deletion
or overwriting. Frozen simulator/proposal files remain untouched.
