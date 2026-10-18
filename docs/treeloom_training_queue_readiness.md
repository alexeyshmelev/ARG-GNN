# TreeLoom full-training queue readiness

This note describes reusable training code and the fresh-campaign adapter. It does not authorize or launch a job. Paths below are repository relative. Historical products retain their original parent identities.

Implemented: `src/arggnn/benchmarking/treeloom_fresh.py`, with 10 passing synthetic CPU tests in `tests/test_treeloom_fresh.py`. The adapter changes private registry/type bindings while reexecuting verified numerical source unchanged. No actual parent preparation, model construction, training, prediction or CUDA allocation was performed for these tests. The full-training runner and first substantive execution remain separate work.

## Reusable numerical implementation

Use `src/arggnn/benchmarking/treeloom.py` as the maintained numerical facade:

- `build_model()` constructs the selected C3 H050 model on CPU, with seed 70001, FP32, 49 parameter tensors and 32,132 parameters.
- `prepare_observation(state, *, resource_check, **kwargs)` delegates complete observed-parent preparation.
- `train_step(model, bundle, targets, optimizer, metadata, *, resource_check, profile=None)` performs one complete-parent update through the corrected bounded executor.
- `predict_logits(model, bundle, metadata, *, resource_check, profile=None)` predicts all frozen focals without resolving ancestry truth.

The corrected executor is `research.reviews.TreeLoom_GPU_history_source_20261017_v1.bounded_execution_v1`. Its current facade is numerically reusable for the historical identity domain. It is not yet a fresh-campaign integration. Keep the complete bank on CPU. The facade selects 128-hold history windows. Do not move the whole bank to the GPU.

Load observed inputs with `arggnn.american_c3_gnn_dataset_v1.load_observation(observation_ref, focal_registry_ref, config)`. The C3 configuration requires the actual `native_preparation_ref`, class order EUR/EAS/AFR, 1,600 haplotypes, reference counts [200, 200, 200], and null `input_reference_class_mask` and `input_reference_sample_indices`.

The fast preparer is `research.reviews.TreeLoom_preparation_reuse_source_20261017_v1.optimized_preparation_v1.prepare_from_loaded_observation_v1`. Its existing complete-parent call supplies `split="train"`, actual observation/focal/native-preparation refs, `max_nodes=400000000`, `max_holds=100000`, `max_episode_payload_bytes=8*2**30` and the admitted worker's `resource_check`. These limits reject oversized input, rather than subsample it.

## Training loop to reuse

`research/reviews/TreeLoom_GPU_history_source_20261017_v1/worker_v1.py` contains the reviewed numerical loop:

- Fresh seed 70001 and AdamW with learning rate 0.001, weight decay 0.0001, betas (0.9, 0.999) and epsilon 1e-8.
- Thirty passes over twenty complete parents, giving 600 genuine optimizer updates and all 10,000 focal targets per parent.
- Gradient clipping at norm 1 and checks of all 49 optimizer counters after every update.
- Checkpoints at epochs 5, 10, 15, 20, 25 and 30, with the final epoch 30 fixed for selection.
- No validation or test ancestry truth used for training or checkpoint selection.

Reuse the numerical loop, not its historical controller, admission or accounting shell.

`epoch_order_v1(epoch)` in `research/reviews/frozen_benchmark_training_ready_20261017_v1/american_full30mb_train_v1.py` copies the declared training-parent list and shuffles it with `random.Random(70002 + epoch)`. Apply this same ordering rule to the explicitly frozen fresh parent list. Do not call that file's `run_train_v1` unchanged. It transfers the whole bundle to the device and uses the older complete-parent executor.

Full-parent GPU fit and throughput remain unmeasured for the fresh campaign. Measure the first real update during substantive training. Another FP64 parity campaign is not needed.

## Fresh-campaign integration

The historical `TrainTargetJoinV1` in `research/reviews/TreeLoom_GPU_history_source_20261017_v1/targets_v1.py` hard-pins the inactive parent registry at `research/reviews/ARGformer_American_C3_implementation_20261015_v1/parent_seed_registry_inactive.json`, SHA256 `608da56508757646a0584cb653f7f6af4c910f05d560d6da1b99c0c1ea691f9a`. It accepts only `american_c3_train_000` through `019`, or the one-parent technical condition. It cannot consume the fresh `argformer_american_train_*` campaign as written.

`FreshTrainTargetJoin` implements the flexible TRAIN-only target join, bound to the fresh immutable parent registry and actual observation, focal, native-preparation and truth refs. It preserves:

1. Authentication that observations and truth belong to the same simulated parent.
2. Frozen focal identities and coordinate support, actual query sample identities and half-open coordinate lookup.
3. Complete eligible support without purity or majority filtering.
4. Separation of ancestry truth from observed features, independent parent splits and explicit C3 class order.

Three further historical identity restrictions are handled by the fresh maintained adapter:

- `optimized_preparation_v1.py` lines 46–47 accept only literal historical TRAIN/VAL IDs.
- `AmericanTreeLoomWholeParentV1.validate` in `american_full30mb_frozen_v1.py` lines 94–98 repeats the same historical membership requirement.
- `bounded_execution_v1.train_step_v1` lines 230–231 requires membership in the historical `frozen.TRAIN`. Its executor also requires exact frozen model and bundle types.

The observation loader and focal registry compare actual parent identities, while these preparation, bundle and execution boundaries additionally hard-code the historical domain. Replacing only the target join will therefore not make the fresh queue runnable.

`FreshTreeLoom` supplies isolated private namespaces for preparation, bundle validation and execution, bound to one fresh immutable registry. It preserves the numerical function bodies, complete-parent support, resource failstops and sample/class constraints. The private model, bundle and executor share matching exact types. Historical pinned implementations and original module globals remain unchanged.

The registry schema is `treeloom_fresh_parent_registry_v1`, with `split_unit=independent_simulation_parent`, class order EUR/EAS/AFR, sequence length 30,000,000 and explicit parent rows containing `parent_id`, `split` and a unique `simulation_stream`. Splits are `train`, `validation` and `test`. Fresh IDs are never renamed. Validation and test share the immutable numerical source's internal observation-only `val_prediction` phase. The maintained API retains the true split and provides no held-out truth resolver.

The fresh full-training runner can use `FreshTreeLoom.build_model`, `prepare_observation`, `train_step` and `predict_logits`, with `FreshParentRegistry.epoch_order` for the unchanged ordering rule. It must still supply actual immutable parent inputs and admitted worker callbacks. Keep the numerical objective and selected architecture explicit. This is data integration, not an architecture contribution. Do not rename old products to satisfy a new registry.

## Existing old-condition products

The targeted completed-metadata census found the following old products. It did not read scientific payloads or perform a general filesystem search.

| Original parent | Simulation | Inferred observation | Native 10,000 focals |
|---|---|---|---|
| american_c3_train_000 | Completed | Completed | Completed |
| american_c3_train_001 | Completed | Completed | Not evidenced |
| american_c3_train_002 through 004 | Completed | Not evidenced | Not evidenced |
| american_c3_train_005 through 019 | Not evidenced | Not evidenced | Not evidenced |

The old twenty-parent condition therefore still needs 15 simulations, 18 inferred observations and 19 native focal preparations. These counts do not describe readiness of the fresh campaign. “Not evidenced” means no completed metadata in the inspected original named result namespaces.

TRAIN000 observation: `experiments/results/American_C3_observed_full30mb_supervisor_repair_20261015_v1/observed/manifest.json`, SHA256 `c3d58a8d10d901a162f406e2e218a10e17e5f1617b3c626503d72bcbd1b4ccee`.

TRAIN000 native preparation: `experiments/results/American_C3_native_paths_train000_full30mb_seed92001_v3/completed.json`, SHA256 `3cc49fbf55bd4b4b3859a93ce1123e9218f407f7c7299ad5029be81f59c7c6fa`. Its `focals.npz` SHA256 is `853b28cf2383d4e7fdf5f2c24526c64ef8071ce81ec04a07d1d69cd7350c8da5`. The metadata records exactly 10,000 query focals, reference-only contrastive labels and no opened query ancestry truth. Label-blind first-occurrence midpoints are a disclosed reconstruction assumption.

TRAIN001 simulation: `experiments/results/frozen_TreeLoom_full30Mb_TRAIN001_root02_oct17_v5/completed.json`. Inference: `experiments/results/frozen_TreeLoom_full30Mb_TRAIN001_infer_root01_oct17_v2/completed.json`, with `observed/manifest.json` SHA256 `b785c856b4426a20f1a5df0951ed9043371cf012e16d85784c2a3197bfd2ff31`.

TRAIN002–004 simulation directories are `experiments/results/frozen_TreeLoom_full30Mb_TRAIN_002_root01_oct17`, `TRAIN_003_root01_oct17` and `TRAIN_004_root01_oct17`, each with `completed.json`.

These products describe 30 Mb parents with explicit haploid interpretation, 1,600 haplotypes, 600 pure references and 1,000 queries. Simulation manifests contain sanitized oracle topology, which cannot substitute for inferred trees. The TRAIN000 and TRAIN001 inference manifests identify tsinfer/tsdate reconstruction. Historical inactive all600 templates are not completed inputs.

## Scientific interpretation

A genuine paired rerun shares the actual genotypes, sample roster, coordinate map, inferred genealogy, reference/query assignment and frozen scoring support. Each method may retain its native representation and training regime. ARGformer representation learning and vocabulary fitting are specific to its observed ARG. Token identities must not be silently reused across independent ARGs.

Twenty supervised training parents are our declared design, not an exact recreation of an unavailable paper split. Haploid counts, JC69/discrete mutation, inference preprocessing and focal-coordinate reconstruction are material disclosed settings. Agreement with a printed table is not the criterion for a valid new paired rerun.

C3 with 32,132 parameters, C4 with 32,325 parameters and Ancestral Paths native six-class output with 32,391 parameters are task-dependent widths of the same selected H050 trunk. They are not separate architecture discoveries. Switching from physical-span loss to sampled-example loss changes the objective and protocol, and must be reported as such.

Existing short-parent training followed by 30 Mb prediction does not establish full 30 Mb training. Preserve those results under their original conditions. No completed fresh comparison or first-place claim follows from readiness.
