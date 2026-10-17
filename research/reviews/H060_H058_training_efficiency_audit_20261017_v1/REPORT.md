# H060/H058 source-only training efficiency audit

Date: 2026-10-17. Repository: `/workspace-SR003.nfs2/ARG-GNN` on `anogena`.

The narrow first successor is H058 CPU metadata-view reuse. H060 also has a concrete startup duplication that can be removed without changing its scientific objective. Larger disjoint-tree batching is a later engineering step. These are source-supported opportunities, not measured bottlenecks or speedup claims.

## Evidence and scope

The actual admitted inputs are `H060_isolated_CPU_ROOT_actual_tx_20261017_v1/input.json` and `H058_CPU_ROOT_tx72_20261017_v1/input.json`, both beneath `research/reviews/`. Their transaction workers select the H060 isolated CPU runtime worker and H058 full-parent CPU runtime wrapper v2, respectively. Their scientific bindings select:

- H060: `research/reviews/H060_isolated_hold_compatibility_source_20261017_v1/warmstart_train_v2.py`.
- H058: `research/reviews/H058_full_parent_CPU_execution_source_20261017_v1/producer_v1.py`.

`admitted_source_receipts.v1.json` records exact metadata refs and 12 H060 plus 11 H058 selected scientific source hashes. All 23 selected files matched the actual outer admitted closure's pins. This establishes the inspected lineage; it does not claim that every transitive source file was individually reviewed. `source_locators.v1.json` records function locations.

Only source/config/admission metadata was read. No scientific payload, truth, runtime file, startup marker, log, checkpoint, process/resource state, or guard state was read. No training, numerical probe, install, Git operation, or modification of existing source/admission/job occurred.

Both selected workers are CPU-only, FP32, 32 threads and one interop thread. H060 retains 20 original complete 2Mb TRAIN parents, 1,600 haplotypes, 1,000 queries, 600 references and 10,000 native focals, with 600 additional complete-parent updates per arm. H058 retains 32 original complete 2Mb TRAIN parents, 952 samples, 686 queries and 266 references, with 960 complete-parent updates per arm. No proposed change reduces those scopes, the two learned tree passes, gradients, sites, updates, or original objectives.

## 1. H058 repeatedly reconstructs immutable CPU index views

In `research/reviews/biological_operator_next_fresh_20261016_v1/packed_csr_v2/same_marker_packed_csr_v2.py:99`, `PackedHoldCPUV2._transient_device_view` converts eleven int32 vectors to int64 and recreates the reference-group one-hot matrix. CPU `.to(dtype=torch.long)` still allocates and copies. The helper runs for the hold gate at line 463, for every marker block at line 402, and again when checkpointed computation is recomputed. These arrays depend on observed layout, roster and route, not learned parameters.

Construct immutable CPU int64 views once during the existing packed preparation/cache stage, with authenticated identity/version stamps. Reuse them for the gate, every marker block and checkpoint recomputation. Store no gates, upward states, evidence messages or learned activations. Keep the current non-CPU path separate; this audit conveys no GPU authority.

For a connected hold with V nodes, Q queries, R references, H samples and G groups, the additional persistent view bytes are exactly
`8*(5V+2Q+3R+H-2) + 4*R*G`, before object/stamp overhead. For the source-known illustrative binary-tree dimensions V=1903, Q=686, R=266, H=952, G=8, this is 109,592 bytes per hold. This is not a measured corpus total. Charge every retained allocation and object in the existing parent/aggregate cache accounting and retain all current limits.

## 2. H060 duplicates observation preparation during startup

The actual runner already has a complete-parent cache. In `warmstart_train_v2.py:416`, it seals all 20 providers, globally prepares and admits all 20 bundles, then crosses the all-parent observation barrier before external TRAIN targets. Later paired updates reuse `_cache_get_v2` (line 234). The 96GiB `max_cached_parent_bytes` allowance is distinct from the zero-byte legacy block cache. It would be incorrect to infer 600 repeated full preparations from `CPU_observation_cache_bytes=0`.

The avoidable duplication is earlier. The inherited `src/arggnn/american_c3_gnn_stream_train_isolated_v2.py:131` seal calls `_read` on every block, hashes each result, then releases the dataset state. The isolated provider's `__getitem__` (`observation_provider_v1.py:15`) constructs native trees and encodes their support. Next, `warmstart_train_v2.py:142` loads the same observation again and the isolated full preparation rebuilds tree/support arrays globally.

Process each parent through its original seal and global preparation while retaining one observation-only loaded state. Reuse once-computed native layout/support primitives where practical, admit its complete global bundle, then release temporary state. Finish every parent's original seal and global preparation before any target access. Preserve original block hashes, focal coverage and complete-parent identities.

Do not assume that arbitrary slices of globally packed arrays reproduce legacy block bytes: local offsets, level arrays and block-specific allele-axis width can differ. Sharing per-tree observed primitives is the safer seam. This removes startup work; it does not eliminate learning-time computation.

The duplicated primitive is substantive: `src/arggnn/mutation_routed_tree_observations_v1.py:262` loops sites, allocates descendant allele counts and traverses postorder nodes in Python before branch discrepancy reductions. Its work scales approximately with the sum of sites × nodes × allele width. A future level-batched preparation kernel must preserve native unary branches, ties, missing calls and isolated sampled roots.

## 3. H058 can share arm-independent topology preparation

`H058_full_parent_CPU_execution_source_20261017_v1/observations_v1.py:59` already loads all original banks with two CPU DataLoader workers, builds the original episodes once, freezes them and prepares both arm routes before targets. `payload_v1` at line 178 returns the frozen packed cache. There is no reason to add another generic cache or repeat admission/hashing every update.

However, `_prepare_payload_once_v1` at line 145 invokes `prepare_packed_parent_v2` separately for each arm. The differing input is the reference sequence `route_permutation`; topology, sample/query joins, reference groups and level indexes agree. A successor can prepare immutable common tensors once and create two small route wrappers with distinct permutations, stamps and bindings. Shared storage must be charged once, each distinct permutation and wrapper charged honestly.

The CPU F1 binding also compares an already supplied CPU bank to that same bank. An explicit same-object-and-version fast path can establish equality without scanning equal bytes, while retaining original authentication and mutation checks (`observation_binding_v3.py:37,184`). This is a secondary startup saving.

## 4. Batch the added same-marker route only after the narrow change

The base GNN is already forest-batched across height/depth levels (`src/arggnn/anonymous_clade_gnn_v1.py:90,110,130`). The H058 added route is different: `integration_hooks_v3.py:66` dispatches one hold at a time; `same_marker_packed_csr_v2.py:342` then runs upward/downward positive-log reductions for that hold and marker block.

Bounded batches of disjoint complete holds could use global node offsets and combine equal-level operations. Pad marker slots only as masked non-observations, keep each hold's query/reference/permutation reductions separate, and scatter back to the original native rows. Chunk under the existing memory limits; retain every real site and complete-parent loss. Complexity remains proportional to total sites × nodes × groups × allele width, and padding can add work. Larger CPU operations may improve utilization, but source inspection cannot quantify that benefit.

Nested marker-block, whole-hold and backbone checkpoints are deliberate activation-memory tradeoffs. Do not simply disable them. The H060 stable-leaf memory recurrence also has a sequential hold loop; exact affine-scan/fused evaluation is a secondary candidate, retaining both directions and full gradients. Avoid exponentially growing prefix rescaling.

## Handoff and interpretation

First implement a new H058 CPU-view successor, then independently consider shared topology and the H060 startup seam. Keep architecture research separate from these efficiency fixes. Validate immutable bytes/joins, original outputs and all learned gradients in a separately authorized successor workflow, then qualify complete-parent memory and actual timing before admission.

Startup markers do not establish an optimizer step. H060's selected worker writes `started.json` at line 83 before invoking the runner; H058 writes `ROOT_started.json` at line 84 before context construction and observation loading. The parent-supplied live observations therefore cannot identify either current scientific phase from source. This audit makes no current-phase or causal-delay claim.
