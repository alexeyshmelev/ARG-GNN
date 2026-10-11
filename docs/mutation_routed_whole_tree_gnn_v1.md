# H036 direct mutation-routed full-parent discovery

This executable source/config implements the registered H036 initial budget. Source authoring and bounded CPU checks do not authorize a launch or imply an accuracy result. Root admissions, reservations and launcher bindings remain separate.

Each worker fits one arm, aligned or shuffled, with seed101. The frozen width32 model has29,764 parameters. Both workers construct the same CPU initialization before CUDA transfer and report its content fingerprint. The only arm-specific model input is the placement of observed branch-support vectors.

The driver takes one optimizer step per complete TRAIN parent, two steps per epoch and10 epochs:20 updates per arm. AdamW uses learning rate0.001, weight decay0.0001, betas0.9/0.999, epsilon1e-8, and gradient clipping1. The two original TRAIN parents share the deterministic order random.Random(101+epoch). Twenty parent updates expose the same10 parent epochs as160 historical250kb tile updates, with a different optimizer update unit. Any later continuation requires an explicit root freeze; this driver stops at20.

Every original observed marginal interval, all280 sampled haplotypes, native internal/unary branches and full bidirectional genomic history remain. No artificial tile reset or history detachment is applied. No emitter, teacher, frozen-logit input, query-age centering or zero residual head is used.

## Observation, target and scoring order

The observed-only loader prepares all four parents and freezes sham geometry bins once from the two TRAIN parents. It returns CPU banks and original scoring metadata. The driver writes observations_features_ready.json before opening either TRAIN target array. Original C4 interval targets enter only unchanged exact_span_loss over all120 actual queries.

The model emits one logit vector per native interval/haplotype, then gathers the existing fixed readout centers and query roster. It covers genuine observation tails that contain no scored bin. Scoring metadata retains the exact original clipped bins, which need not start/end at inferred-coverage limits.

Each epoch produces raw FP32 softmax probability seals for both VAL parents. The driver writes the all-VAL-sealed manifest before calling unchanged H025 score_v1 and the original external truth/scoring helper. Zero probabilities remain zero; no floor or posthoc normalization is applied. Checkpoint selection uses the earliest strict minimum equal-parent Present64 exact-span raw-probability NLL. All120 and ancient/cohort diagnostics remain secondary.

Only owned superseded probability seals/checkpoints are pruned. Current last and selected best checkpoints/seals remain. The worker output cap is1,000,000,000 bytes. Observation byte snapshots use output.path/observation_tmp, with automatic file deletion after verified tskit loading, keeping scratch inside that worker's reservation. No persistent large feature cache is written.

The first actual full-parent TRAIN forward/backward/update is step1 of20, rather than a dummy or disposable qualification fit. Its receipt records synchronized forward/backward/optimizer time, CUDA allocated/reserved peaks, CPU maximum RSS, native holds/vertices, all280 haplotypes, sites, query columns and readout counts. Runtime process naming and CUDA context admission use the existing configure_fit helper.

## Files and invocation

- Neural model: src/arggnn/mutation_routed_whole_tree_gnn_v1.py, frozen separately.
- Preprocessing: src/arggnn/mutation_routed_tree_observations_v1.py, frozen separately.
- Loader: src/arggnn/mutation_routed_parent_loader_v1.py.
- Driver: src/arggnn/train_mutation_routed_whole_tree_gnn_v1.py.
- Config: configs/mutation_routed_whole_tree_gnn_v1.json.
- Author receipt: research/reviews/mutation_routed_whole_tree_gnn_v1_author/driver_receipt.json.

The existing full-storage launcher dispatches this module with:

~~~text
-m arggnn.train_mutation_routed_whole_tree_gnn_v1
--execute
--input <root-pinned arm input path>
--expected-input-sha256 <root-pinned input digest>
~~~

Without --execute the module prints source-only information and performs no preparation, fit or GPU operation. The root launcher owns physical GPU selection, lease, title and reservation. Each worker must receive its own fresh exclusive output directory and storage admission.

## Root input skeleton

This is a template, not an executable authorization. Root fills fresh timestamps, refs, source hashes and observed-parent bindings. Copy the original H035 observed_parents and target_bindings structures, then remove emitter/teacher requirements. No original array reads are needed to compose the capability.

~~~json
{
  "schema": "mutation_routed_whole_tree_gnn_input_v1",
  "status": "active",
  "root_authorizer": "root_agent",
  "fit_authorized": true,
  "job_id": "<fresh distinct arm job>",
  "arm": "aligned",
  "seed": 101,
  "device": "cuda",
  "epochs": 10,
  "created_at_utc": "<fresh UTC timestamp>",
  "expires_at_utc": "<fresh UTC timestamp within 24 hours>",
  "max_cpu_rss_bytes": 128000000000,
  "max_cuda_reserved_bytes": 68719476736,
  "output_directory": "<fresh reserved project output>",
  "refs": {
    "config": {
      "path": "configs/mutation_routed_whole_tree_gnn_v1.json",
      "sha256": "<config hash>"
    },
    "runtime": {"path": "<pinned runtime>", "sha256": "<hash>"},
    "storage": {"path": "<fresh arm reservation>", "sha256": "<hash>"},
    "target_bindings": {"path": "<original C4 binding>", "sha256": "<hash>"}
  },
  "source_sha256": {
    "src/arggnn/train_mutation_routed_whole_tree_gnn_v1.py": "<driver hash>",
    "src/arggnn/mutation_routed_whole_tree_gnn_v1.py": "<frozen model hash>",
    "src/arggnn/mutation_routed_tree_observations_v1.py": "<frozen preprocessing hash>",
    "src/arggnn/mutation_routed_parent_loader_v1.py": "<loader hash>"
  },
  "observed_parents": {
    "bio_paths_pilot_train_000": "<same original observed refs object>",
    "bio_paths_pilot_train_001": "<same original observed refs object>",
    "bio_paths_pilot_val_000": "<same original observed refs object>",
    "bio_paths_pilot_val_001": "<same original observed refs object>"
  }
}
~~~

The shuffled worker changes arm, job/output/storage/timestamps and admission digest; scientific config, seed, original observations/targets and model source stay identical. Config helper_refs pin unchanged scoring/loss/lease/full-storage sources. The root source closure may also include those helpers.

## Scope and interpretation

The candidate uses original observed genotypes and qualified tsinfer/tsdate trees. Frozen native ARGMix uses Relate genealogy. These share the underlying physical observations and scoring queries, but do not form a matched-topology architecture comparison. The trainer never loads old teacher arrays. Existing native/smoothing results belong in a separate truthful comparison after both workers finish.

Whole-tree supervision, cavity sweeps, gates, genotype/genealogy fusion and recurrence are established ideas. The narrow mechanism claim concerns correct placement of observed mutation-concordance support and needs an accuracy advantage over the placement sham. Genuine routing must also beat ordinary smoothing to meet the current goal. A sham tie or NLL-only improvement does not establish that mechanism. Reused development VAL is not final independent confirmation.

The author receipt records source/config parsing, identical seed101 initialization, raw-probability seal schema, strict earliest-minimum selection, observation-before-TRAIN-truth order, both-VAL-before-scoring order and owned seal pruning. No actual observation arrays, GPU, scientific optimizer update or native fit was executed during authoring. Frozen model/preprocessor sources, budgets, ledgers and Git were not modified.
