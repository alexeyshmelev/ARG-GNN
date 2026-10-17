# H060 source throughput diagnosis and inactive exact legacy candidate

Source-only review package. No execution/admission authority, launcher, model
step, truth reader, persistent feature cache, installation or Git operation.
The currently admitted/live sources and work remain unchanged.

## Finding

The supplied 15-second observation (about 0.63 core on the main thread, most
other threads asleep, no sampled I/O) is consistent with serial preparation.
It does not identify a Python frame or prove a whole-run bottleneck. Process
memory profiling was denied; no retry, sudo or alternative memory access was
attempted.

The concrete source path explains why the fast-global change cannot remove
all serial work:

- `H060_fast_observed_full_training_source_20261017_v1/warmstart_train_v2.py:441–451`
  completes each of the original 20 parents sequentially before TRAIN truth.
- `TreeLoom_observed_preparation_fast_source_20261017_v1/parent_startup_fast_v1.py:79–93`
  loads once, then calls the original provider's **legacy seal before fast
  global preparation**. Fast global preparation has not begun during that seal.
- The original seal in `src/arggnn/american_c3_gnn_stream_train_isolated_v2.py:138–144`
  calls `_read` for every full physical block, which calls the original H060
  dataset at `H060_isolated_hold_compatibility_source_20261017_v1/observation_provider_v1.py:24–30`.
  Connected blocks dispatch to `src/arggnn/american_c3_gnn_observations_v1.py:75–84`.
- `src/arggnn/mutation_routed_tree_observations_v1.py:280–305` loops over every
  site, then every postorder node at 291–293, then every proper allele split.
  The per-node NumPy row additions are Python-dispatched. This path is still
  present even though the subsequent global pass uses Torch/PyG level sums.

Other serial source work remains: native-tree conversion repeats per-block and
again globally (`observations_v1.py:49–54`, `full_preparation_fast_v1.py:66–72`),
`_layout` builds/walks Python child lists (`mutation_routed_tree_observations_v1.py:180–199`),
clade preparation traverses every original node and constructs exact bitset
clades (`anonymous_clade_flag_control_v1.py:41–85` and the episode builder), and
cut preparation visits every node/sample plus episode triples
(`order2_cut_memory_v1.py:191–274` and compatible isolated wrapper). Global
tensor hashing also reads all bytes. These are candidates, not measured shares.
No claim follows that the CPU is I/O-bound, that 32 threads are ineffective in
learned tensor work, or that any particular speedup/runtime has been proved.

## Concrete inactive remedy

`legacy_support_exact_v1.py` replaces only the legacy inner per-site node
sweep with the already reviewed Torch/PyG level-batched **int64** descendant
counts. It preserves original site order, proper allele order, all native
branches, tied winners, NumPy float64 per-split summary additions and the final
FP32 cast. The float64 operations are the existing observed-feature recipe,
not a new precision condition or experiment. Chunk boundaries change only
integer count storage, not floating-point addition order. Original global
FP32 support remains unchanged.

`observations_legacy_exact_v1.py` is a disjoint copy of the isolated-compatible
encoder. It binds this helper explicitly and removes the connected-only early
return so connected blocks cannot silently use the old sweep. Every remaining
function body is AST-identical to that original after removing the dispatch.
`observation_provider_candidate_v1.py` changes only its explicit encoder import.
All original inherited seal/hash/coverage/release methods remain unchanged.

The main count tensor is capped at 8,388,608 int64 elements (64 MiB), using at
most 32 markers per chunk. This excludes scratch, other state and RSS, whose
existing caps remain mandatory. A single marker that exceeds the bound fails
closed. There is no cropping/recompute fallback. Actual full-parent performance
and memory fit are unmeasured. Small synthetic byte-equivalence checks support
only those fixtures; no actual full-parent hash equivalence is claimed.

## Required fresh wiring after independent review

1. Keep all current sources, jobs, outputs, charges, clocks and scientific
   records intact. Do not monkeypatch or migrate a live provider. ROOT owns
   any fresh full-protocol source package, actual accounting and execution.
2. In that new package, bind **both** the warmstart runner's provider import
   and the parent startup helper's provider import/type check to the candidate
   provider. Changing only the runner fails the startup exact-type guard.
3. Retain the single observation load, genuine original seal/hash/coverage,
   complete global fast preparation and genuine newly generated global seal.
   Bind every new helper/encoder/provider/startup/runner source in the complete
   source closure and additional-preparation source refs. Adjust source gates,
   manifests, import closure and runtime qualification accordingly.
4. The startup seal field `legacy_block_seal_producer_unchanged=True` must become
   false with a truthful explicit producer/source binding. This producer's
   implementation changes even when fixture bytes match. Never assert that an
   old actual digest was generated or reuse it without a genuine new seal.
5. Preserve every original 20 complete 2 Mb TRAIN parent, 1600 haplotypes,
   600 references, 1000 queries, 10000 native focals per parent, full topology,
   observed features, structural recipes and source/data validation. Keep all
   20 newly sealed/cached preparations before TRAIN truth. Keep PyTorch/PyG,
   the original accepted saved600 backbone, separate fresh AdamW, both full
   arm models, 600 additional updates per arm / 1200 total, FP32 model math,
   checkpoint order and all current resource/lifecycle guards. No parent
   parallelism or increased concurrency is introduced by this candidate.
6. All 60 new held-out prediction seals must exist before either arm's VAL
   truth or checkpoint selection. TEST/scoring stay deferred. No pilot,
   reduced support, cached truth, automatic retry/resume, old-scope migration,
   new training condition or inferred scientific score follows from this work.

A later optimization could share an observed native topology/layout across
legacy and global work, or parallelize independent parents within the original
aggregate limits. Neither is implemented here because it changes state/memory
ownership and requires separate review. The smallest concrete next source
successor is the explicit exact legacy kernel wiring above.
