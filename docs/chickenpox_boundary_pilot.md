# Frozen Chickenpox boundary applicability pilot

This is an independent fixed-geography applicability pilot of the existing observed-only boundary GNN and memory. It is not a Nikitin graph-SPDE kernel reproduction. The graph is the fixed adjacency of 20 Hungarian counties, so this study measures neither changing-topology generalization nor biological ancestry performance. Its status is implemented and CPU-verified, with no actual model training or benchmark scoring by the implementing agent.

The runner is `src/arggnn/train_boundary_chickenpox.py`. The frozen study is `configs/chickenpox_boundary_pilot_v1.json`, SHA256 `c7bd9bac661a87d6e2852aff8c51863e0543ada46ca5fc9c42ab5a93546af0b4`. It pins the existing common Chickenpox protocol, UCI CC BY 4.0 source archive and source manifest by content hashes. It does not download or modify source data.

## Common observations and chronological protocol

The pinned archive contains 522 weekly observations. Raw case counts remain in the original edge-ID county order. All supervised methods use exactly 52 observed weeks to forecast the following week.

| Quantity | Zero-based inclusive indexes |
|---|---|
| Available training raw observations | 0 to 364 |
| Supervised training targets | 52 to 364 |
| Validation targets | 365 to 442 |
| Reserved final-test targets | 443 to 521 |

The runner masks every numeric row from 443 onward to NaN immediately after archive integrity checking. The masked array is read-only. Only this restricted data object reaches geometry preparation, training-only scaling, history binding, fitting and validation evaluation. Archive byte verification and metadata checks do not score the final-test observations.

Each county's mean and population standard deviation use only rows 0 to 364. Constant training scales become one. No validation observation affects scaling. The supervised training targets are the same 313 forecasts as the existing neural pilot. Every validation bank binds exactly `[target-52,target)`. Thus target 365 uses rows 313 to 364 and target 442 uses rows 390 to 441. Target counts and all later counts remain external to the bank.

The native adjacency has 102 supplied directed records, 20 self loops and 41 unique undirected non-self geographic links. All 20 counties are persistent query nodes. There are no reference nodes or eliminated interior nodes. The response is `C=L`, using one conductance per unique neighbor link and full-boundary mean-diagonal normalization. The raw mean diagonal is 4.1. Native self loops cancel from `D-A`, while the encoder keeps its own residual/self update. Geometry/eigenpairs are prepared once per geography condition and reused across histories. All learned input features are the same training-standardized observed case counts.

Weekly hold coordinates preserve the existing float32 uniform `[0,1]` partition. The readout anchor remains the midpoint of the final observed interval, with `target_graph=51`. The following-week label is external to this historical domain. No actual forecast target is inserted as an additional graph or source hold.

## Predeclared models

All controls have one scalar regression output per county. The graph source encoder uses hidden width 32 and two layers. Memory latent width is eight. The shared boundary model and the geographic adapter remain unchanged. The independent scalar-kernel, GRU and Transformer adapters live in the runner file.

| Model | Parameters | Mechanism |
|---|---:|---|
| `boundary_full` | 4,937 | Full coupled forward memory |
| `boundary_diagonal` | 4,937 | Raw diagonal forward dynamics |
| `boundary_gain_matched` | 4,937 | Full equilibrium gain, independent initial-state dynamics |
| `boundary_gain_leak` | 4,937 | Full equilibrium gain and grounding-row-sum leak |
| `boundary_bag` | 4,937 | Width-weighted homogeneous bag, forward dynamics |
| `boundary_kernel` | 4,418 | Learned positive scalar exponential kernel, left-supported interval mass |
| `boundary_gru` | 12,961 | Unidirectional chronological GRU |
| `boundary_transformer` | 42,785 | Causal Transformer and past-only coordinate cross attention |
| `boundary_full_no_geography` | 4,937 | Full forward memory and the same source encoder with `C=0` |

Full, diagonal, gain-matched, gain/leak, bag and no-geography models have identical state-dictionary shapes. The zero-reference setting leaves 264 reference-projection parameters inactive in each of these models, and this is disclosed in their configuration receipts.

Ordered memory has no backward scan. The scalar kernel integrates only source support left of the historical readout coordinate, including the left half of the last observed hold. Its weights normalize by total permitted kernel mass. The GRU removes the shared control's backward recurrence and adjusts the head to the one-sided context. The Transformer masks record attention to earlier/current records, then masks cross attention to the readout record and its predecessors. Its temporal depth is two, heads four and dropout zero.

The bag control averages all 52 available observed holds and intentionally ignores their chronology. Its homogeneous generator is read with forward duration to the preserved anchor. Its forcing uses the complete observed bag, so it does not satisfy an earlier-anchor causal intervention. This is not future-label leakage, since the whole input bag precedes the external next-week target. The ordered models and the kernel/GRU/Transformer controls are separately checked for zero dependence on later holds at an artificial earlier anchor.

The no-geography condition zeros the response and removes geographic neighbor messages. It preserves county histories, scaling, model parameter schema and forecast anchor. Because this is a small fixed graph and ungrounded response, it is an applicability control rather than evidence about large ARG interfaces.

## Frozen fitting and selection

The runner uses AdamW with learning rate 0.001 and weight decay 0.0001. Each epoch shuffles training forecast indexes under seed 101. Loss is mean squared error in training-standardized county units, gradients clip at norm one, and all validation forecasts are inversely scaled before scoring. There are at most 40 epochs and eight stale epochs of patience. Selection is minimum mean validation MAE in raw case units. Predictions are never clipped, including negative case predictions. Every predeclared model is reported separately.

The CLI permits a distinct subset of the predeclared models for root-controlled scheduling. It provides no test, seed, epoch, optimizer or split override. Each run records its requested subset and whether all predeclared models completed in that particular run. A configuration edit creates a different content hash, and frozen policy checks reject incompatible schedules or boundaries.

## Scheduling and receipts

The root agent owns GPU scheduling. This agent launched no GPU job. The runner-local `initialize_worker` sets the existing project title `AI Auto Research Pipeline Test` before any CUDA inspection. No pre-existing function of that name was found. CUDA runs require the project's eligible physical GPU assignment, matching live lease file descriptor and exactly one visible device. CPU runs require explicitly empty `CUDA_VISIBLE_DEVICES`.

For a CPU invocation, source the project environment and use a fresh output path:

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES= .runtime/env/bin/python -m arggnn.train_boundary_chickenpox \
  --config configs/chickenpox_boundary_pilot_v1.json \
  --expected-config-sha256 c7bd9bac661a87d6e2852aff8c51863e0543ada46ca5fc9c42ab5a93546af0b4 \
  --output experiments/results/chickenpox_boundary_pilot_v1_seed101_cpu \
  --device cpu
```

For GPU scheduling, use `scripts/launch_gpu.py` after verifying an eligible device is free, passing `arggnn.train_boundary_chickenpox` as the module and `--device cuda`. The global two-GPU limit and exclusions remain unchanged. The default CLI device is CPU, so a launcher invocation must explicitly select CUDA.

Each run saves exact copies of the study/common-protocol/source-manifest JSON. Results include all input hashes, starting Git commit, implementation-file hashes, training seed, county order, scaler values, chronological ranges, masked-tail cutoff, runtime geometry manifests and per-model configuration hash/parameter count. The runner checks that implementation hashes remain unchanged through each training run before recording completion.

Checkpoints and compressed prediction arrays stay under project-local `checkpoints/`. Each prediction artifact includes raw predictions, raw validation truth, exact target indexes, dates, county order and study hash. JSON receipts include raw-unit aggregate MAE/RMSE, per-county MAE/RMSE and per-week metrics with historical ranges. Checkpoint and prediction artifacts are independently content-hashed. Training epoch metrics, selected epoch and timing remain recorded. Failed fitting writes a failed progress receipt and never emits a complete results file.

## CPU verification

The dedicated runner suite uses synthetic counts rather than scoring the pinned benchmark. It checks all nine constructors against target/future poisoning, ordered/control outputs and gradients against later-hold poisoning, independent scalar-kernel integral normalization, exact raw-unit county/week receipts, forbidden policy/source changes, training-only scaling, masked-tail delivery before geometry and fitting, unchanged final-observation anchor, title-before-CUDA ordering, unclipped negative predictions and a real small CPU optimization/checkpoint round trip.

The runner and unchanged geographic-adapter suites passed 76 tests together. The final verification log is `.runtime/logs/boundary_chickenpox_runner_tests_v3.log`. The real-source check binds verified observations without labels or numerical benchmark scoring. This verification establishes implementation behavior, not prediction quality.

