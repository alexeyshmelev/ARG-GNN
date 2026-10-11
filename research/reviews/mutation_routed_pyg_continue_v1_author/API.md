# H037 PyG optimization continuation API

H037 is the registered added-optimization condition linked to H036. It resumes each same arm's immutable LAST epoch80 model and full AdamW state, then performs240 additional complete-parent updates, epochs81–200. Primary selection occurs at90,100,…,200 and preserves the strict earliest minimum NLL across initial epochs1–10, prior scheduled12–80 and new90–200. The prospective config has `execution_guard_ready=false`; the root sets it true after review/admission and refreshes its config hash. No model mechanism or superiority is claimed.

Source: `src/arggnn/continue_mutation_routed_pyg_gnn_v1.py`.
Config: `configs/mutation_routed_pyg_continue_v1.json`.

## Root input and admission

Use schema `mutation_routed_pyg_continue_input_v1`, `root_authorizer="root_agent"`, `fit_authorized=true`, `continuation_authorized=true`, `pyg_authorized=true`, `status="active"`, `arm="aligned"` or `"shuffled"`, `seed=101`, `device="cuda"`, `epochs=120`, canonical `job_id`, dedicated fresh `output_directory`, and the original created/expires UTC fields (at most24hours). Supply positive integer `max_cpu_rss_bytes` and `max_cuda_reserved_bytes`.

`refs` contains exact path/SHA256 records for:

- `config`: final admitted config.
- `runtime`: root runtime/environment qualification.
- `storage`: active full-reservation storage record binding source/config/runtime and unchanged real launcher.
- `target_bindings`: original four-parent external C4 truth identities, unchanged.
- `prior_completed`: same-arm completed H036 total160 receipt.
- `prior_input`: the input record named by that prior completion.
- `resume_checkpoint`: the same-arm LAST epoch80 checkpoint named by both prior completion and H037 registration, even when the selected checkpoint was earlier.
- `observation_cache`: root-published shared observed-only manifest.

`observed_parents` remains the exact original four-parent map. `observation_cache_source_refs` is the exact published manifest's source map (root keys `model/preprocessing/loader/efficiency/cache`). The source closure must also include every path in driver `SOURCE_CLOSURE`: new driver/PyG model/PyG Dataset/cache/fast scorer, initial/previous drivers, original model/preprocessor/loader and efficiency helper. Other pinned helper source records can be included.

The config binds the root H037 registration, final new module hashes and unchanged Torch2.6.0+cu124/CUDA12.4 plus PyG2.7.0. Runtime source version changes require a newly reviewed pin. Original title, canonical GPU UUID, lease, own-process NVTOP and real full-storage guards are reused without changes. Each fit scope is capped at500000000 bytes. Root publishes the separate shared cache within its own admitted500MB scope.

Execute only through the root's existing reserved GPU launcher using:

```text
.runtime/env/bin/python -B -m arggnn.continue_mutation_routed_pyg_gnn_v1 --execute --input <root-input-path> --expected-input-sha256 <sha>
```

Calling the module without `--execute` emits only its source/config report and does not create a CUDA context, open actual observations or train.

## Startup and hot loop

Two CPU-only one-thread spawn workers load each complete cached parent once through the PyG Data/Dataset API. Parent items preserve all original native nodes, holds, sample/query indexes and scoring support. Each original worker PID is recorded and explicitly drained after startup, then checked again before completion. The main process obtains original bundle metadata from the verified JSON index without rereading all parent arrays. It durably seals all four completed observed banks before TRAIN truth is opened.

Only the main process creates CUDA state. Before updates, it measures all four banks' exact tensor bytes plus two FP32 TRAIN masses against actual free CUDA memory and the admitted reservation with20000000000bytes workspace retained. If they fit, all four banks and TRAIN masses move once and remain resident. If they do not fit, the job stops before any optimizer update with an admission receipt. No hidden per-update observation decoding or device-copy fallback is used.

The model is `MutationRoutedPyGGNNV1`, which keeps the original38 learned state keys and29764 parameters while using actual parameter-free PyG MessagePassing in the complete-tree sweeps. Strict loading and restored optimizer-state equality are checked once at startup; each trained AdamW counter must be160. No actual-parent precision sweep or dummy optimization update is run.

Each epoch buffers its TRAIN row in memory and prints compact progress. Ten rows are written together at each selection epoch, avoiding per-epoch output/storage guard loops. Lightweight active-capability/lease/CPU/CUDA guards remain around each complete-parent update. Real storage guards remain at startup, evaluation, guarded output writes and completion.

## Validation and retention

Both current VAL raw FP32 probability rasters are durably sealed before `FastPrimaryValidationV1.score` verifies them and accesses any VAL truth. Only Present64 physical-span accuracy and NLL are computed during selection. Its separate process-local score-only target mass never enters observed cache, model input or loss. Original support and zero-probability infinity handling remain unchanged.

At epoch200, final selected sealed predictions pass once through the unchanged complete scorer for class, boundary, ECE, dosage and short-tract diagnostics. This diagnostic pass does not revise selection. The completion links the primary selected record, combined selection history and separate final full-diagnostic artifact. Inherited selected records are external immutable references and are never pruned. Only this output writer's own nonselected probability/checkpoint products are discarded.

Failure preserves the last durable checkpoint, reached logical step count, pending TRAIN rows and undurable update count. It explicitly drains owned data workers. No automatic replay or restart is performed.

## Source-only checks

`receipt.json` records AST parsing, protocol acceptance/rejection, a handmade CPU checkpoint's exact strict state/AdamW160 restore, counter159 rejection, no device copy or CPU decode in the TRAIN function, and one full-diagnostic call outside the epoch loop. The handmade check performs zero optimizer updates and reads no actual checkpoint or observed arrays. Actual cache, CUDA residency, worker lifecycle and throughput remain root-run checks.
