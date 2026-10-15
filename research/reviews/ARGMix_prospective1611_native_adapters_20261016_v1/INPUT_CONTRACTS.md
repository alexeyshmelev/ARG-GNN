# Concrete adapter interfaces

Paths in JSON bundles resolve relative to the project working directory. Numeric artifacts and native dependencies are opened only by explicitly invoked runtime commands. All outputs use exclusive creation; a partial output without complete matching metadata is unusable.

## Roster and mappings

The included `prospective_roster.unbound.v1.json` contains only synthetic observed metadata. Every diploid has phase0/1 with identical cohort/date. Observed IDs must come from the actual Relate conversion, using:

```json
{"schema":"haplotype_node_mapping_v1","rows":[{"haplotype_id":"Ana_0000_h0","node_id":123}]}
```

The real mapping must contain every3,222 haplotype identity exactly once, with distinct nonnegative node IDs. This one-row illustration is not a valid full mapping. Bind separately for every parent. No sample trimming, guessed offsets or mate inference from node arithmetic is allowed.

Simulation truth mapping is a separate document:

```json
{"schema":"simulation_haplotype_node_mapping_v1","rows":[{"haplotype_id":"Ana_0000_h0","simulation_node_id":987}]}
```

It must independently cover the same identities. Simulation population names use `{"schema":"simulation_population_mapping_v1","populations":{"0":"present_bronze","1":"Yam"}}` with every required actual ID/name, verified against simulation population metadata. NF samples map toNeo; Post/Present samples map topresent_bronze; other cohort names retain their simulation population name.

## Common targets and external labels

Each target JSONL row has exactly:

```json
{"row_id":"parent:tree:hap","parent_id":"published_recon_train_000","haplotype_id":"NF_0000_h0","tree_index":0,"left":0.0,"right":10.0,"position":5.0}
```

Coordinates are the exact physical midpoint of the common inferred-tree half-open interval. Targets contain no ancestry label or truth feature. Duplicate target IDs/pairs, conflicting intervals for one tree index, nonquery identities and nonmidpoints fail. Actual observed sample nodes/times and target tree intervals are checked during extraction. Generator/root owns the frozen sampler (TRAINi%10=0, VALi%100=0, all IID/OOD trees; seed registry and uniform100k support from the predecessor plan).

Labels are a separate JSONL containing exactlyrow_id/ancestry. `truth_adapter.py` constructs them once from the simulated census350 and the common target coordinates. The command emits a mandatory `labels.jsonl.contract.json` binding:

- Parent, label/target/roster/observed-tree hashes and target count.
- Exact census350 founder definition and constructor source hash.
- Simulation tree, separate simulation mapping, population mapping and demography hashes.
- Explicit demography with zero BAA samples, admixtures younger than350 and founder divergences older than350.

The constructor rejects missing census350 ancestors; no rounding-to350, cohort-identity fallback, majority-interval target, or observed-node-equals-simulation-node assumption is used. A future truth constructor run and its actual truth inputs must still be sealed by root. Fit-time contract validation checks the emitted metadata and does not independently reopen biological truth payloads.

Both methods' supervised input bundles must point to the same labels and contract for each parent. Fit summaries include labels_sha256, supervision_contract_sha256 and truth_definition_sha256. `paired-fit-summaries` rejects different labels/truth provenance even when target IDs are identical. Prediction input bundles omit labels and contracts entirely.

## Feature bundles

The included inactive templates name five TRAIN parents, one VAL parent and three IID parents. TRAIN/VAL entries have exactlyparent_id/roster/targets/features/metadata/labels/supervision_contract; prediction entries omit the last two. AP and ARGMix feature paths differ; all shared observation/target/truth identities agree.

Feature extraction takes observed trees, bound roster and target metadata only. ARGMix retains all observed interval context for queried haplotypes. C4 AP emits only target feature rows. Both feature metadata files pin the common tree/roster/target bytes, source/config and output bytes. `paired-metadata` verifies their shared observation support.

Fit requires exactly5×100k registeredTRAIN targets and1×100kVAL targets, with disjoint parents and target IDs. It imports native algorithms only after metadata/contract validation. This source task does not establish a peak-RAM/storage/runtime admission; native full-context materialization needs root-owned accounting.

## Empirical map

`map_binding.template.inactive.json` is incomplete by design. A real binding needs explicitGRCh37/chr22/public-sourceURL/sourceSHA/source interval/rebased fileSHA/L. The local rebased map has the exact header:

```text
position COMBINED_rate(cM/Mb) Genetic_Map(cM)
```

Positions are strictly increasing, span exactly0throughL, cumulativecM starts0 and is nondecreasing; rates/distances are finite/nonnegative. Both fit and prediction reject feature intervals outsideL, preventing the native interpolator's edge extrapolation. Root must independently establish public source/build and the exact interval/rebasing derivation: the adapter verifies local bytes/coverage, not public provenance. No actual map or guessedL is included.

## Output classes

Both prediction streams return one row per common target in target order, with labels[Ana,CHG,WHG,EHG] and four normalized probabilities. C4 AP uses pooled native probabilities before four-way argmax, while ARGMix has a four-class native head. Focal query/cohort tokens are not ancestry labels. Native Rust labels are discarded; only the external census constructor supplies supervision.
