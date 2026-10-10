# Canonicalized observed-tree GNN+GRU comparator

This dedicated train/validation implementation addresses the canonicalization comparator requested in `research/protocols/confirmatory_design_v1.md`. It uses the existing inferred Q64 cohort with 16 TRAIN, four VAL and no TEST parents. These are reused development parents, not the proposal's fresh development or independent confirmation cohorts. No actual-data training, scoring, GPU job, installation, Git operation or shared source edit was performed for implementation. Root owns scheduling and subsequent inference.

## Observed preprocessing and correspondence

`prepare_canonical_parent(observed_sequence, reference_adapter, base_feature_count=8, ...)` accepts an observed sequence and fixed options only. It receives no target tensor, dataset object or truth tree. It first strips appended summaries to the original F8 prefix, invokes the existing observed-tree canonicalizer with anonymous unary suppression and **exact adjacent duplicate merging enabled**, and then recomputes `ReferenceFeatureAdapter` on the complete canonical observed parent. The result has the same F48 field schema as the source adapter, with the canonical F8 prefix retained first. No stale source summaries are reused.

The parent-branch field is recomputed from decoded retained endpoint ages and re-encoded by the existing utility. Observed reference roles, query/leaf flags, encoded ages and retained descendant-landmark indicators otherwise remain available under their established contracts. IDs and recursive matching signatures are correspondence keys, never numerical learned features. Labels, source proportions, populations, migration records, census vertices and truth trees are not consulted by preprocessing.

`CanonicalPreparedParent` contains the augmented sequence, `CanonicalizedObservedTrees` correspondence and adapter provenance. Its weights-only-safe dictionary contains observed inputs and audit/correspondence fields, with no labels. `source_graph_to_canonical`, source graph groups and representative source node indexes remain available for review. Scoring positions, their dtype/order, query identities and external query order are retained. The containing graph is remapped explicitly.

`canonical_fixed_tiles(parent, windows=8)` clips the canonical parent into the same fixed global-coordinate windows and observed SNP coverage. Each `CanonicalTile` carries explicit original parent position indexes. These indexes gather external training targets and scatter validation predictions; concatenating chronological tiles is not used. This preserves unsorted original readout order and never extrapolates uncovered chromosome tails. The pure tile factory receives observed inputs only. External targets are joined afterward by the training wrapper and used only for loss/metrics.

`prepare_canonical_gnn_input(sequence)` caches stable leaf/query node lookup, graph chronology, inverse ordering and physical width/midpoint-gap tensors once on CPU. Transfer `CanonicalGNNInput` with `.to(device)` before a model call. The GRU does not perform repeated GPU-side query matching.

## Ordinary model and honest scope

```python
from arggnn.canonical_gru import (
    CanonicalTreeGNNGRU, prepare_canonical_parent,
    prepare_canonical_gnn_input,
)

parent = prepare_canonical_parent(observed_sequence, reference_adapter_options)
model_input = prepare_canonical_gnn_input(parent.sequence)
model = CanonicalTreeGNNGRU(
    input_dim=48, num_classes=4, hidden_dim=32,
    num_layers=2, temporal_layers=2, dropout=0.,
)
logits = model(model_input)
```

The GNN uses the existing residual mean-message layers on **all canonical tree nodes** and supplied bidirected topology. It has an input projection followed by two graph layers. Stable query leaves are gathered in external query order. Their graph-local embeddings enter a conventional two-layer bidirectional GRU, together with each physical interval width and previous midpoint gap. Widths/gaps are computed in original global chromosome coordinates, with float64 differences before feature-dtype conversion. The first gap is zero. A linear head gathers the containing canonical graph for each requested readout and returns `[P,Q,4]`.

Dropout applies after input projection and each GNN layer, between stacked GRU layers, and before the output head. Evaluation disables dropout. This definition is recorded explicitly rather than assuming an unused `dropout` setting in the legacy GRU. The new GRU has two temporal layers, whereas the older `GNNGRU` helper had one. This is an independently tuned ordinary comparator rather than an exact paper reproduction.

Full-tree F48 includes the recomputed canonical parent-branch field and internal-node information. The boundary candidate uses persistent leaves and F47 with parent-branch excluded. This source-scope distinction is allowed by the confirmation proposal and must be disclosed. This comparator is not a strict same-source boundary mechanism control. Landmarks remain canonicalization/matching metadata and are not concatenated as additional learned features in this mean GNN.

At source depth two and GRU depth two, widths32/64 have **37,988 / 145,604** parameters. Dropout and learning rate do not change counts. These exceed the candidate's parameter count, and no capacity equality is asserted. Predictions are constant within a canonical hold for a fixed query at evaluation, because readouts gather the same recurrent graph token. Additional coordinates do not add temporal source tokens.

The representation contracts arise from standard exact preprocessing: unary suppression, duplicate merging, deterministic canonical tensors and stable query correspondence. Deterministic consumers of identical tensors share these contracts. This is **not novel invariance**, a new kernel, or evidence of predictive superiority. Training comparisons with dropout require the same canonical tensors and random state; deterministic evaluation is the reported contract.

## Hash-bound pilot and fair family design

`configs/canonical_inferred_cohort64_pilot_v1.json` pins inferred manifest `611c7e32973c22bd35115b835279884f7b0c866136e91ae2c5f2a95daefd6fe7` and dataset protocol `66c6bb30191013d5c2a16a40ca870d313c858eb5f05cb3afc9aa88ed50514cb5`. It preserves source adapter settings, windows8, 64 query haplotypes, four class slots and the 16/4/0 parent split. The dedicated runner reuses the existing read-only source loader's raw/prepared/metadata hash, prefix, adapter, seed and split checks. It does not edit `train_study.py`, its registry or any shared source.

The proposed family has exactly eight configurations:

| Axis | Frozen values |
|---|---|
| Hidden width | 32, 64 |
| Dropout | 0, 0.1 |
| Learning rate | 0.0003, 0.001 |
| Source GNN / GRU depth | 2 / 2 |
| Training seeds | 101, 202, 303 |
| Epoch cap / patience | 48 / 8 |
| AdamW weight decay / gradient clip | 0.0001 / 1 |

Each configuration is initialized independently from its prescribed seed. Training uses cross entropy on external targets and shuffled tiles, with parents split before any tiling. Select a checkpoint by minimum equally weighted **mean parent validation NLL**, keeping the earliest epoch on an exact tie. No test records or nonprescribed seeds/configurations are accepted by the CLI.

Primary NLL comes from original raw logits in float64 stable log-softmax, through root's pure `confirmatory_metrics.logit_metrics`. It has no probability floor. Compatibility metrics retain the older probability-floor NLL separately. The raw metric details also include confusion/support, classwise NA denominators, fixed-zero all-class macroF1, Brier and15-bin raw calibration/class reliability. No temperature calibration is fitted by this runner. Diploid dosage is not reported without an explicitly supplied haplotype pairing map.

`select_family_configuration(records, settings, expected_config_sha256, config_bytes=original_bytes)` requires all eight configuration × three seed records before selecting a family. The expected hash must come from the original config bytes or supplied freeze, never from reserialized JSON. It rejects missing/duplicate starts, every protocol hash that differs from that exact expected config SHA, record specifications or parameter counts that differ from the frozen models, and mismatching dataset, implementation/dependency or raw/prepared/metadata source hashes. It averages the three selected-checkpoint validation NLLs per configuration. Configurations within0.0001 nats of the minimum prefer fewer parameters, then lexicographic ID. An individual seed run or partial grid remains an incomplete family study. No refitting on VAL or extra successful seeds are supplied.

## Dedicated runner

`python -m arggnn.train_canonical_gru` accepts `--config`, optional `--expected-config-sha256`, `--data`, a fresh `--output`, `--seed`, optional frozen `--configurations` IDs, `--device cpu|cuda` and `--prepare-only`. CPU is the safe default and requires an empty CUDA-visible environment. CUDA use requires the existing project launcher lease and one allowed reserved device. Root performs all GPU scheduling.

A CPU preprocessing-only command for root's later use is:

```bash
. scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$ARGGNN_PYTHON" -m arggnn.train_canonical_gru \
  --config configs/canonical_inferred_cohort64_pilot_v1.json \
  --expected-config-sha256 89de3bcf5cbb15015c6e13b1ae9770f4ef3f89c9a4533b185289a8e69debbf30 \
  --data data/ancient_inferred_cohort64_pilot_v1 \
  --output results/canonical_cohort64_preparation_v1 \
  --seed 101 --device cpu --prepare-only
```

This command was **not executed** during implementation. Preparation writes observed canonical input/correspondence caches and per-parent source/adapter receipts inside its new output directory. Training mode additionally saves checkpoints, original-order validation logits with position/query IDs, configuration histories and per-parent metrics. The output records dataset/protocol hashes, source/dependency hashes, code commit, environment, physical GPU assignment, preprocessing/training time and peak CUDA allocation. No final-test path is opened or test cohort generated.

Important dependencies include the canonicalizer, reference adapter, genealogy helper, data/models, source loader, metric readers, resource guards and training helpers. The config binds their exact reviewed source hashes under `implementation_sha256`; a missing or mismatching binding aborts before outputs. The config also pins all raw/prepared/metadata source artifact hashes against its verified dataset manifest. The runner captures exact config/manifest bytes and their hashes before observed preprocessing, honors a supplied `--expected-config-sha256`, and copies those captured bytes into the output. A pre-read guard rejects confirmatory, confirmation, final_test, final-test and test namespaces in config, manifest, source and output paths before opening their bytes, including resolved symlink destinations. Configuration IDs and parent cache IDs must be bounded ASCII filename components; slashes, backslashes, dots, traversal, external absolute names, empty/oversized values and duplicate split/parent IDs are rejected before derived paths are formed.

Config, manifest and implementation/dependency hashes are checked at preparation/configuration boundaries and before/after each epoch, including before saving a newly selected checkpoint. Raw/prepared/metadata source artifacts are fully SHA256-verified before and after preprocessing and each configuration; their inode/device/size/mtime/ctime signatures are also checked through preparation and each epoch. Full boundary hashing additionally catches changed bytes even if a stale filesystem stat is reported. Run records use the initial snapshots, record their source-artifact bindings and describe the checks. A change aborts the run; refreshing reviewed bindings requires a new frozen run version. The model manifest reports source scope, layer/dropout definitions, parameter count and implementation hash.

## CPU verification and limitations

The final comparator/canonicalizer/raw-metric integration passed **90 tests in10.95 seconds**, including all46 standalone canonical comparator/runner tests. The CLI help check also passed with the expected-config-hash option.

Checks include this new model's own predictions after combined unary insertion, exact duplicate splitting and node storage permutation, at dropout0/0.1 with deterministic evaluation and deliberately replayed training RNG. They verify branch/prefix/fresh-adapter schema, explicit graph/readout correspondence, poisoned truth attributes, external-label counterfactual independence, a source-label counterfactual across references with different query TMRCAs, stable ID/query/readout permutation, added-readout independence, unsorted parent-coordinate scatter, finite graph/GRU/head gradients, synthetic five-epoch CPU fitting/checkpointing, unclipped extreme-logit NLL, full grid/budget binding and seed/capacity/lexicographic selection guards. Adversarial runner checks reject unsafe configuration/parent/source path values and duplicate cache IDs, enforce supplied config/reviewed code/source hashes, preserve captured bytes, reject midrun config/manifest/code/source mutations, and stop an injected epoch-boundary mutation before checkpoint saving. Regression tests reproduce and reject auditAD01 held-out read attempts before any bytes are opened, and auditAD02 arbitrary common protocol hashes/wrong record specifications/capacities during family synthesis.

A pinned **training** rep000 observed prepared input was also read without its raw target bundle. Canonicalization/fresh augmentation preserved F48, all64 queries and original coordinates. A finite untrained first-tile output had shape `[16,64,4]`. No actual labels, optimizer fitting or scores were used in this smoke test. Existing cohort parents may already have no unary or duplicate records, so canonicalization need not reduce their size or improve predictions.

These results establish implementation/correspondence and small CPU trainability. They do not complete the 24-start family search, establish independent confirmation, or show a biological/predictive benefit. Root's later fair comparison must preserve complete learning curves, failures, preprocessing costs and the disclosed source/capacity differences.

## Admission-order reaudit repair

Root moved the canonical runner's device admission after complete config, observed-source and selected-ID admission. Invalid or held-out inputs now reject before any CUDA availability/device-count query. Two CPU and CUDA-request monkeypatch cases demonstrate this ordering without querying a GPU. The dedicated canonical suite passed48 tests in13.06s on the repaired source. The new pilot config SHA is89de3bcf5cbb15015c6e13b1ae9770f4ef3f89c9a4533b185289a8e69debbf30, and the runner SHA iscdb85ef8a031e472c2ba7844074670307542f94636e58cfa482a17aa672618ac. No actual canonical fitting was performed. The earlier90-test union included supporting suites as well as this dedicated suite.

The subsequent independent replay found a coherent caller-settings drift: mutated settings and all24 consistent records could carry an unchanged original config hash. The helper now requires captured original config_bytes, hashes those exact bytes and checks parsed-settings equality before record/model work. The accepted pre-repair reproduction remains preserved. The dedicated suite now passes49 tests in12.16s including off-grid width16 under unchanged original hash rejection. Fresh development uses a separate completion-bound adapter and the generic immutable selector.
