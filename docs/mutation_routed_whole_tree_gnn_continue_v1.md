# H036 exact continuation to 160 full-parent updates

This separate driver resumes each arm's **last epoch-10 checkpoint**, including every AdamW moment, counter and parameter group. It does not restart training, choose an earlier best checkpoint for resumption, introduce a new arm/seed, alter the frozen neural model, or detach genomic history.

The initial runs completed20 updates per arm. Root authorized continuation after their last four evaluated epochs still improved NLL, with majority-class accuracy unchanged. The continuation adds70 full-parent epochs,140 updates, giving80 global epochs and160 logical updates per arm. Added optimization exposure is explicit and cannot be described as the initial equal-exposure comparison.

## Cadence and selection

Every global epoch11–80 performs one update on each of the two complete TRAIN parents. Parent order is random.Random(101+global_epoch), continuing the original epoch index. Every completed TRAIN epoch receives a durable log.

VAL prediction, scoring and model checkpoint writes occur at global epochs:

~~~text
12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80
~~~

Both VAL probability rasters are sealed before either VAL truth opens at those epochs. Intervening epochs use cached TRAIN supervision and do not open VAL truth. Selection remains the earliest strict minimum Present64 equal-parent raw-probability NLL, among all ten initial evaluations and the eighteen scheduled continued evaluations. No accuracy reselection or unscheduled candidate is permitted. Terminal epoch80 is always evaluated and saved.

The configuration explicitly changes evaluation cadence from the initial every-epoch protocol. That change applies equally to both arms and must be registered before extra updates.

## State and provenance

The continuation verifies the completed initial receipt, initial input, exact original observations/target bindings, last checkpoint identity, frozen initial configuration and complete source closure. It constructs the original module/optimizer objects, then overwrites all trained parameters and optimizer state from the epoch10 snapshot. Exact restored state equality is checked. The seed101 initialization content fingerprint remains the original fingerprint.

Initial best checkpoint and prediction records remain immutable external references. They are not copied into the new output or passed to its discard method. The combined selected row may remain an initial row if all later scheduled NLL values are worse. Newly written last/best checkpoints and seals belong only to the continuation writer.

The efficient observation helper has the same original preparation API and bundle. It changes one quadratic null-constant diagnostic assignment to an exact grouped implementation while preserving other preparation code, RNG, arrays and receipt semantics. Its independent function globals do not mutate frozen modules. Separate efficiency provenance identifies the helper and original sources. All original native holds, all280 samples, genotype evidence, sham placement, physical scoring bins and query identities remain.

Full observations are durably sealed before TRAIN target arrays open. Verified tree byte snapshots use the new worker's output/observation_tmp and auto-delete after loading. No teacher/emitter cache or persistent large feature cache is introduced.

## Failure and recovery

Only scheduled evaluation checkpoints are durable model/optimizer states. Every TRAIN log identifies the previous durable checkpoint and updates completed since it. A failure receipt records the last durable epoch/checkpoint, logical steps reached and uncheckpointed updates. There is no automatic replay or resume.

Recovery would need a separately registered exact continuation from that durable snapshot. Any re-executed unsaved updates must be disclosed in execution accounting. Earlier initial or continuation outputs are never edited by this driver. Full-parent history remains intact inside every update.

The output cap is750,000,000 bytes per worker. The first actual full-parent update was already measured in the initial run. No saved dummy update or extra disposable profiling fit occurs during continuation.

## Files and root capability

Driver: src/arggnn/continue_mutation_routed_whole_tree_gnn_v1.py
Config: configs/mutation_routed_whole_tree_gnn_continue_v1.json
Helper: src/arggnn/mutation_routed_observation_efficiency_v1.py
Receipt: research/reviews/mutation_routed_whole_tree_gnn_continue_v1_author/receipt.json

Root launches through the existing reserved full-storage launcher using:

~~~text
-m arggnn.continue_mutation_routed_whole_tree_gnn_v1
--execute
--input <new arm continuation input>
--expected-input-sha256 <new input digest>
~~~

Without --execute, the module is source-only. The following input skeleton is not an executable authorization. Root fills fresh refs, timestamps, job/output/reservation and hashes.

~~~json
{
  "schema": "mutation_routed_whole_tree_gnn_continue_input_v1",
  "status": "active",
  "root_authorizer": "root_agent",
  "fit_authorized": true,
  "continuation_authorized": true,
  "job_id": "<fresh distinct continuation arm job>",
  "arm": "aligned",
  "seed": 101,
  "device": "cuda",
  "epochs": 70,
  "created_at_utc": "<fresh UTC timestamp>",
  "expires_at_utc": "<fresh UTC timestamp within 24 hours>",
  "max_cpu_rss_bytes": 128000000000,
  "max_cuda_reserved_bytes": 68719476736,
  "output_directory": "<fresh exclusive reserved output>",
  "refs": {
    "config": {
      "path": "configs/mutation_routed_whole_tree_gnn_continue_v1.json",
      "sha256": "<config hash>"
    },
    "runtime": {"path": "<pinned runtime>", "sha256": "<hash>"},
    "storage": {"path": "<fresh continuation reservation>", "sha256": "<hash>"},
    "target_bindings": {"path": "<same original binding>", "sha256": "<hash>"},
    "initial_completed": {"path": "<same-arm initial completed.json>", "sha256": "<hash>"},
    "initial_input": {"path": "<same-arm initial input>", "sha256": "<hash>"},
    "resume_checkpoint": {"path": "<same-arm LAST epoch10 checkpoint>", "sha256": "<hash>"}
  },
  "source_sha256": {
    "src/arggnn/continue_mutation_routed_whole_tree_gnn_v1.py": "<new driver hash>",
    "src/arggnn/train_mutation_routed_whole_tree_gnn_v1.py": "<frozen initial driver hash>",
    "src/arggnn/mutation_routed_whole_tree_gnn_v1.py": "<frozen model hash>",
    "src/arggnn/mutation_routed_tree_observations_v1.py": "<frozen preprocessing hash>",
    "src/arggnn/mutation_routed_parent_loader_v1.py": "<frozen loader hash>",
    "src/arggnn/mutation_routed_observation_efficiency_v1.py": "<efficient helper hash>"
  },
  "observed_parents": "<exact original initial-input object>"
}
~~~

The shuffled worker uses its own original completed/input/last checkpoint and new job/output/storage refs. Scientific configuration and source closure remain identical. Fresh root continuation authorization is required; an old initial admission cannot launch this driver.

## Bounded verification and scientific limits

Author checks used a synthetic in-memory checkpoint only, with nonzero AdamW moments and step20, without an optimizer update. Model and optimizer state restored exactly. The initial earliest NLL tie remained selected, imported records stayed external,20+140 equalled160, and the scheduled28-candidate history ended at epoch80. Source checks verified observation sealing before TRAIN truth, both VAL seals before scoring, strict NLL selection, own scratch scope and unsaved-update reporting. No actual checkpoint, observed array, GPU or scientific fit was opened during authoring.

The original genotype/genealogy information difference versus Relate-based native ARGMix remains disclosed. Reused development parents and increased training exposure do not establish independent generality or a novel GNN mechanism. Predictive conclusions await the paired continuation outcomes.
