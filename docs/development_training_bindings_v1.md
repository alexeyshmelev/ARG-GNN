# Bound fresh-development temporal baselines v1

This adds independent development adapters only. Existing preparation code, its freeze/configs and shared model implementations were not edited. No simulation, native inference, GPU launch, model-fitting run, package operation or Git action was performed by this task.

The new files are:

- src/arggnn/development_training_bindings.py
- src/arggnn/train_anchor_temporal_development.py
- scripts/bind_development_training.py
- tests/test_development_training_bindings.py
- docs/development_training_bindings_v1.md

## Completion-to-config boundary

The binder defaults to **validate-only**. It reads the successful existing preparation completion receipt at experiments/results/fresh_development_preparation_v1/completion.json, checks 64 TRAIN / 16 VAL / TEST 0, all-stage success, all 640 inferred cache tiles and the distinction between preparation and training/confirmatory selection. It verifies all four manifest SHAs, source-parent → inferred → boundary → age relationships, exact parent identities, simulation/inference masters 20261101/20261102, and factual preparation hashes. It calls the existing full boundary and age cache loaders to verify source files, cache envelopes and numerical/partition contracts. It rechecks the receipt and stage manifests afterward. No target-based filtering or final-test path is supported.

--write-configs also requires an explicit completion-file SHA and a new destination under repository configs. No placeholder manifest/preparation hashes are permitted. The destination is not created when completion is absent or verification fails. The default completion currently does not exist, so no executable configs were emitted during this task. Validation and config generation are root's next steps after successful preparation.

The generated candidate family is compatible with root's train_anchor_development.checked_development_protocol and contains exactly eight plain/age × kernel-depth1/2 × LR0.0003/0.001 configurations, width32, source-depth2, latent8 and fixed 5/20/80 coupling branches.

Each temporal family gets eight configurations: width32/64 × dropout0/0.1 × LR0.0003/0.001. Both use plain BoundaryResponseGNN depth2, latent8, temporal-depth2, four Transformer heads, 32 physical midpoint anchors and three source-rate branches. Ordinary decay is0.0001 and source-rate/mixture decay is0. Both use epochs48, patience8 and seeds101/202/303. Actual GRU/Transformer capacities and parameters remain disclosed by their existing model manifests.

The smoother gets eight bandwidth10/30/100/300kb × pseudocount0.001/0.01 configurations. It uses the existing observed reference-role/plain-Cqr evidence model with no fitted parameters. It chooses hyperparameters using raw equal-parent validation NLL. Three seed runs produce the same deterministic predictor; they do not add independent training or parent evidence.

## Exact runner binding and artifacts

The new runner requires --expected-config-sha256 at startup, captures the configuration bytes once, validates their hash and requires the file to remain identical after parsing. It writes those captured bytes to protocol.json and uses that one fixed hash for every checkpoint, prediction and result. It does not recompute a changed live hash and silently associate it with old in-memory settings.

The runner rejects namespace/count/seed/budget/grid/model-ID drift, checks bound implementation hashes against imported source snapshots, checks the completion receipt's config/cache facts, and verifies actual completed inputs before training. It checks config/receipt/header/code hashes and all artifact stat identities during epochs. Full byte hashes for the complete input catalog are checked at startup, before/after each configuration and at run seal, without reloading cohort tensors per epoch. Final-test paths are unavailable. Inputs come from config-bound development cache paths; there is no arbitrary --data override.

Neural checkpoints select the minimum uncapped float64 log-softmax NLL equally averaged over independent VAL parents, with the earliest epoch on exact ties. Metrics join external labels only after model forward and exact parent coordinate/query scattering. Arbitrary target readouts never become temporal sources. The existing physical-coordinate decoder and common observed-source contracts remain unchanged.

All .pt checkpoints and raw validation-logit snapshots live under ignored checkpoints/temporal_development_v1 with output-identity, config-hash and seed namespaces. Tracked run outputs contain captured protocol, compact dataset binding, histories, parent metrics, model manifests, source/config hashes and paths/hashes to snapshots. An output or checkpoint namespace already present is rejected. Numerical/model failures retain a failure.json receipt; bank-transfer failures also have a phase-specific receipt. There is no auto-resume, test-best checkpoint/seed choice, replacement family or hidden retry.

Family synthesis and the strongest-baseline choice remain separate. Complete the prescribed three-seed neural searches and retain every failed run before selecting a family on development VAL. This adapter does not promote a validation result to confirmation.

## Root commands after preparation completes

Run on anogena within /workspace-SR003.nfs2/ARG-GNN and source scripts/project_env.sh first. The default check is:

~~~bash
CUDA_VISIBLE_DEVICES='' "$ARGGNN_PYTHON" scripts/bind_development_training.py
~~~

After computing and reviewing the completed receipt's SHA, root can generate the four new config files in a fresh dedicated directory:

~~~bash
CUDA_VISIBLE_DEVICES='' "$ARGGNN_PYTHON" scripts/bind_development_training.py \
  --expected-completion-sha256 RECEIPT_SHA \
  --write-configs \
  --output-directory configs/development_training_bound_v1
~~~

RECEIPT_SHA is a documentation argument, not a placeholder written into an executable config. The tool prints actual config SHA values and writes a binding_receipt.json listing the families.

Before scheduling, a baseline family config can be checked on CPU with its printed actual SHA:

~~~bash
CUDA_VISIBLE_DEVICES='' "$ARGGNN_PYTHON" -m arggnn.train_anchor_temporal_development \
  --config configs/development_training_bound_v1/gru.json \
  --expected-config-sha256 CONFIG_SHA --seed 101 --validate-config
~~~

Use the same module for transformer.json and reference_smoother.json. Root may choose --models from the immutable eight-config family to distribute jobs while preserving the full search. Neural fitting through scripts/launch_gpu.py uses --device cuda and a fresh --output, with seeds101,202,303. The smoother can use CPU. Candidate.json uses root's existing train_anchor_development module, --data and --age-data from the completion binding, and the actual printed config SHA. No fitting commands were executed here.

## Validation and limits

The final integration check passed **60 CPU synthetic tests in6.70s**: the new binding/config-immutability/grid suite plus existing temporal runner/model suites. It includes safe IDs, exact hash rejection, configuration edits during/after capture, candidate compatibility, complete source/cache loader calls, completion-fact mismatches, no-write default binder, held-out path refusal, label counterfactuals, independent readouts for all three baseline types and finite neural gradients. Handmade GraphSequence fixtures are used; no chromosomes are simulated and no native Relate job is executed.

The first new-suite run had five failed checks. It exposed shared mutable template constants that could make an in-memory config edit mutate its validator's expected counts/seeds. Family templates now deep-copy counts/seeds, with a regression check for isolation. The negative duplicate-grid fixture was also corrected to duplicate both the specification and learning rate. The initial failures and corrected test commands are retained in the task record.

CLI help parsed successfully. The default binder returned failure because a successful preparation completion is not yet available and wrote no configs. Therefore actual 64/16 cache binding and real optimizer runs remain unexecuted. Initial/ending hashes and synthetic contracts establish implementation safeguards, not empirical ancestry benefit, novelty or a final-test freeze.

Root's current preparation freeze continues to govern data generation and admission. Full cache verification is CPU- and RAM-heavy, especially eager age loading, and is repeated for actual runs. Later GPU training follows the project launcher, eligible physical devices2–5, the two-GPU cap and storage guard. Nothing here changes those policies.


## Independent audit fixes AD-03 and AD-04

The independent canonical/development audit correctly found that the earlier
live check omitted inferred/oracle raw artifacts and the actual preparation
freeze, and that the candidate runner did not consume generated completion/code
locks. Those findings remain retained in
research/reviews/canonical_development_audit_v1/review.md.

Both candidate and temporal runners now require the same mandatory
admit_run entrypoint. The candidate consumes the completed receipt, exact bound
cache paths, actual preparation freeze, reviewed dependency map and imported
code snapshot before loading usable training inputs. Supplied data/age paths
must equal the bound paths. Held-out config, completion, source and supplied
data paths are rejected before reads. Every stage manifest must match its real
freeze directory before opening it. Immutable original config bytes/SHA remain
the identity for protocol copies, checkpoints, predictions and results.

capture_completion_sources verifies the actual freeze file, kind/access flags,
the recomputed complete80-parent ancestry/mutation/Relate registry, frozen
preparation configs/code and every bound input artifact. It covers oracle and
inferred metadata/raw/prepared tensors, original truth and observed archives,
inferred observations/native trees/execution/input-export artifacts, external
supervision, plain banks, age sidecars and verification receipts. Truth
genealogies are hashed as bytes and are not decoded or passed to models.

SourceSnapshot records SHA256 and device/inode/size/mtime/ctime identity. Epoch
checks inspect file identity plus small config/receipt/header/code hashes and
do not reload the80-parent tensor cohort. Configuration and run seals fully
rehash the catalog and recheck stat identity before/after hashing. Changed or
absent input files prevent complete result sealing even when the already-loaded
tensors themselves remain unchanged. Compact dataset bindings record the source
snapshot, and completed run records preserve its ending artifact hashes.

Safe explicitly listed extra .py dependency pins are supported beyond common
CORE for separately owned comparator adapters. Common dependencies cannot be
removed, and every supplied dependency path/hash is verified. This does not
authorize edits to active pilot/core/preparation files.

Final focused integration verification passed104 CPU synthetic tests
in96.85s, including the candidate's one-parent tiny fitting fixture with only
cohort admission mocked. Production admission has no bypass. Adversaries
mutate oracle/inferred manifests, raw tensors, metadata, genotype exports,
banks, sidecars, supervision, recipe and freeze files. They also cover a
rehashed wrong freeze registry, absent/wrong/incomplete completion, a stage
path mismatch before file reading, held-out access before reads, missing
candidate locks and same-size edits with restored mtime. The original source
manifest mutation pattern now raises instead of returning a successful live
check. These safeguards do not establish ancestry benefit or final confirmation.

No real fresh-cache admission, production fitting, simulation, native
inference, GPU, Git or package action was run during this fix. Active preparation
and pilot dependencies remained untouched.


## Device admission ordering follow-up

Independent reaudit found that the temporal development runner called
require_device before admit_run. Its CUDA branch can query device_count, and
its CPU branch queries is_available. The temporal runner now completes
mandatory cohort/config/dependency/source admission before require_device.
The candidate runner already had this ordering and was checked without edits.
Early process-title setup remains unchanged.

Four dedicated ordering regressions in
tests/test_train_anchor_temporal_development.py cover CPU and CUDA requests.
Actual missing completion is refused before any device check or patched CUDA
query, and successful mocked admission reaches the device check afterward.
No CUDA context, production fitting or data generation is used by these tests.
The focused check passed4 tests in5.48s. The expanded binding, candidate,
temporal development, temporal pilot and temporal model CPU regression suite
passed108 tests in126.70s. Only the temporal development runner, its dedicated
new tests and this task's evidence documents changed for this follow-up.
