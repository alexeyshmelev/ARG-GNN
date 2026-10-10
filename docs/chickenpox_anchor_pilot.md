# Direct fixed-anchor Chickenpox validation pilot

This runner applies the unchanged `AnchorKernelModel` directly to the same observed weekly county histories used by the existing Chickenpox pilot. It does not substitute the predecessor's one-sided continuous-forcing memory model for the new architecture. It is an independent applicability study on fixed county geography, not a Nikitin model reproduction, an ancestry experiment or a claim of novelty or predictive superiority.

The implementation is `src/arggnn/train_anchor_chickenpox.py`. The frozen config is `configs/chickenpox_anchor_pilot_v1.json`, SHA256 `f72fdfd0bf1aa148df0e291421513c4269883b9df43c42eeda0ea26a5ecbbeb7`. The implementing agent launched no GPU jobs and performed no actual benchmark fitting or scoring. Small CPU fits in the tests use synthetic counts.

## Observed history and forecast boundary

The runner pins the existing common Chickenpox protocol, the original UCI archive and its source manifest by content hashes. There are 522 weekly observations in edge-ID county order, with 20 counties. Its rolling one-week-ahead forecasts use exactly the same boundaries as the prior neural applicability pilot:

| Quantity | Zero-based inclusive indexes |
|---|---|
| Raw observations for training-only scaling | 0 to 364 |
| Supervised training forecast targets | 52 to 364 |
| Validation forecast targets | 365 to 442 |
| Reserved final-test target rows | 443 to 521 |

Every history for target `t` contains only `[t-52,t)`. Thus there are 313 supervised training forecasts and 78 validation forecasts. Validation is rolling observed-history forecasting. The first validation target uses weeks 313 to 364, and the last uses weeks 390 to 441. Earlier validation counts can appear in a later validation history because they have been observed by that later prediction time.

Immediately after archive integrity and metadata checking, the runner replaces all numeric rows from 443 onward with NaN in a read-only restricted data object. Only that masked object reaches training-only scaling, geometry preparation, feature binding, fitting and evaluation. The complete archive's bytes and dates are verified as source metadata, without final-test numerical scoring.

Per-county mean and population standard deviation use only raw rows 0 to 364. A constant training scale becomes one. Standardized features have the exact same float32 values as the existing observed-history adapter. Labels remain outside every bank.

## Direct architecture adaptation

The constructor is the actual core `AnchorKernelModel(input_dim=1, num_classes=1)`. Its graph encoder, normalized kernel messages, source projections, nonlinear gates, branch mixing, repeated contextual depth and linear output head remain unchanged. The output head predicts a standardized scalar county count, which is inversely scaled into raw case units for validation.

All counties are persistent queries, with no reference or interior nodes. The fixed geographic interface is `C=L`, with one conductance per unique undirected non-self native neighbor link and full-boundary mean-diagonal normalization. The original adjacency has 102 directed records, 20 supplied self loops and 41 unique geographic links. Self loops cancel from `D-A`. The source GNN has its own self and residual updates.

Each history has 52 fixed physical midpoint anchors across its normalized observed domain. There is one source anchor per observed week. The anchor layout, three repeated plain geometry branches and cached eigenpairs are prepared once per geography condition and reused across all histories. History binding replaces only observed count features. Sources always stay on the 52 fixed anchors. The readout does not become an additional source.

The readout retains the original float32 midpoint of the final observed weekly hold, with `target_graph=51`. Fixed anchors are computed in float64, so the final readout and the final source anchor can differ by rounding. The core evaluates that exact readout coordinate using its completed kernel. The external next-week forecast label is never represented as an input graph, source anchor or observed feature.

The three branches repeat the same plain geographic operator. They initialize coupling rates at 5, 20 and 80, with damping one, and have learned branch mixtures. There are no biological age operators in this pilot.

## Completed history context

All 52 weekly observations precede the forecast target and are available to its prediction. The unchanged core therefore uses completed bidirectional kernels throughout that available history. Contextual depth can transmit later observed history into an earlier historical anchor before the final decoder reads the historical anchor coordinate.

This is a history encoder followed by an external next-week decoder. It does not assert internally causal updates at an earlier history anchor. The tests positively verify that the first history anchor receives information from the last available observed week. Separate future-poisoning and raw-count gradient tests verify that observations at or after the external forecast target cannot affect its output.

## Frozen models and active parameters

The source encoder has width 32 and depth two. Latent width is eight, anchor count 52 and branch count three. All specs use global log-parameterized rates, with `rate = initial * exp(raw)`, and softmax branch mixtures. D1 and D2 mean one and two nonlinear kernel layers.

| Spec | Coupling | Kernel depth | Total parameters | Structurally active parameters |
|---|---|---:|---:|---:|
| `anchor_full_d1` | Full coupled | 1 | 6,617 | 6,353 |
| `anchor_full_d2` | Full coupled | 2 | 8,849 | 8,321 |
| `anchor_diagonal_d1` | Raw diagonal | 1 | 6,617 | 6,353 |
| `anchor_diagonal_d2` | Raw diagonal | 2 | 8,849 | 8,321 |
| `anchor_gain_matched_d1` | Gain-matched diagonal | 1 | 6,617 | 6,353 |
| `anchor_gain_matched_d2` | Gain-matched diagonal | 2 | 8,849 | 8,321 |
| `anchor_scalar_d1` | Scalar time kernel | 1 | 6,617 | 6,353 |
| `anchor_scalar_d2` | Scalar time kernel | 2 | 8,849 | 8,321 |
| `anchor_full_d2_no_geography` | Full with `C=0` | 2 | 8,849 | 6,225 |

Empty reference inputs make 264 projection parameters per kernel layer structurally inactive. In the full no-geography condition, `C=0` also makes 48 coupling-rate parameters and 2,048 source-GNN neighbor weights structurally inactive. Tests verify their zero gradients. The active counts exclude those explicit blocks. They do not estimate effective degrees of freedom or count parameters with a nonzero gradient on a particular batch.

No-geography preserves observed histories, county order, scaling, source encoder schema, completed kernel depth and last-observed readout. It removes geographic neighbor responses. The scalar family uses the unchanged core's `K=(lambda+d)I` control. The gain-matched family keeps the unchanged source/mass transformation and its documented finite-anchor scope.

Static/local and noncontextual controls are not part of this frozen v1 protocol. If integrated later, they require a separately recorded expanded protocol rather than silently changing the nine-model comparison.

## Fitting, validation and diagnostics

Training uses seed 101, AdamW, learning rate 0.001 and ordinary parameter weight decay 0.0001. All coupling, damping and mixture parameters are placed in a separate zero-weight-decay group, matching the current direct-core study. Training shuffles the 313 forecast indexes per epoch and minimizes standardized county MSE. Gradients clip at norm one.

Every model has at most 40 epochs and patience eight. Selection uses minimum chronological validation MAE in raw case units. All raw predictions remain unclipped, including negative predictions. The CLI allows only distinct subsets of the frozen model list for scheduling. It has no test, optimizer, epoch, seed or split override.

Each epoch records finite pre-clipping coupling/damping/mixture gradient norms, current rates/damping/mixtures and maximum relative rate/damping changes. Missing or nonfinite kernel gradients cause a failure. Initial and selected-checkpoint model manifests are saved separately, so completion receipts describe learned selected rates rather than initialization alone.

Results preserve protocol/source/configuration hashes, starting Git commit, all relevant direct-core and helper dependency hashes, county order, scaler values, masking cutoff, chronological ranges, geometry metadata and structural parameter counts. Dependency hashes are checked again after fitting, and a changed implementation cannot be recorded as a completed frozen run.

Raw prediction archives contain exact validation target indexes, dates, county order, raw predictions and raw validation truth, along with the study hash. Model receipts contain aggregate and per-county MAE/RMSE plus per-week metrics and each historical range. Checkpoint and prediction arrays are content-hashed and remain under project-local `checkpoints/`. A failed fit or changed implementation writes a failed progress receipt without a completed results file.

## Scheduling

Only the root agent launches GPU jobs. The runner reuses the verified predecessor's `initialize_worker`, which sets `AI Auto Research Pipeline Test` before CUDA inspection, then checks the eligible physical device, project lease descriptor and single visible device. CPU runs require explicitly empty `CUDA_VISIBLE_DEVICES`.

For a CPU invocation with a fresh output directory:

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES= .runtime/env/bin/python -m arggnn.train_anchor_chickenpox \
  --config configs/chickenpox_anchor_pilot_v1.json \
  --expected-config-sha256 f72fdfd0bf1aa148df0e291421513c4269883b9df43c42eeda0ea26a5ecbbeb7 \
  --output experiments/results/chickenpox_anchor_pilot_v1_seed101_cpu \
  --device cpu
```

A GPU launch uses `scripts/launch_gpu.py` with `arggnn.train_anchor_chickenpox` as its module and explicitly passes `--device cuda`. The existing two-GPU limit and GPU exclusions apply. GPU defaults are not selected by this module.

## Verification

The dedicated tests use synthetic counts, apart from reading pinned source bytes and metadata. They verify actual core constructors and forward passes, all 52 observed sources, exact preserved readouts, target exclusion, future-poisoning invariance, zero gradients to raw counts at/after the forecast target, permitted bidirectional known-history context, reused static geometry/layout, masked reserved rows before downstream operations, source/config/policy pins, raw per-county/week prediction receipts, current selected-state diagnostics, zero-decay rate groups, structural inactive blocks, negative unclipped outputs and a real small CPU fit/checkpoint round trip.

The dedicated suite passed 56 tests. Combined verification of the new runner, predecessor runner, unchanged geographic adapter and actual anchor kernel passed 211 tests in 10.45 seconds. The final verification log is `.runtime/logs/anchor_chickenpox_tests_v2.log`. This verification establishes implementation behavior and direct architectural applicability. It does not establish prediction quality or scientific superiority.
