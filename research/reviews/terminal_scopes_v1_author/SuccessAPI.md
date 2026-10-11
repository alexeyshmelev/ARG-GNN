# Narrow terminal success successor

Frozen source: `src/arggnn/project_storage_terminal_success_scopes_v1.py`, SHA256 `2dfbcc2711fc8310fd8a8b9866ee84e9107e1d2c4356396e0ebc0e99fdb7c3b3`.
Mode: `journal_terminal_success_scopes_credit_v1`. Journal is the SAME existing `.runtime/storage/project_budget_terminal_scopes_v1`. No new journal layer and no changed historical helper/events.

The previous terminal v1 events are a validated immutable prefix. New events have the same append-only event schema/sequence/hash chain, with new successor bindings and operation `close_completed_cpu_observed_absent` or `reserve`. Any v1 event appearing after a successor event fails closed. Root must complete/drain/revoke every v1 terminal writer before the first new event. The currently admitted v4 cache producer and VAL006/007 remain on their frozen v1 guard until they finish. The successor still replays original CPU/GPU success proofs and all failed-cache credits unchanged.

Public APIs retain `inspect_v1`, `apply_reserve_v1`, `budget_check_v9`, `validate_storage_v9`, `PilotOutputV9`, and `check_storage_locked_v1`; the locked function does not acquire the mutex. New closure API is `apply_close_completed_cpu_observed_absent_v1(authority_ref, expected_source_sha256=CORE_SHA, expected_launcher_sha256=BASE_SHA)`.

Root bindings are the previous four `implementation`, `launcher`, `cpu_overlay`, `terminal_overlay` (frozen terminal v1 ref), plus `terminal_success_overlay` (new successor ref). Initial/expected ledger pins remain the same five keys: legacy/supplemental hashes, original journal tail, CPU overlay tail, and shared `terminal_overlay_tail_sha256`. All five source refs must be included in new reservation consumer bindings, along with the original v9 consumer stub and actual source/config/runtime/launcher dependencies.

Root authority requires `kind=project_budget_terminal_success_root_admission_v1`, `root_authorizer=root_agent`, `active=true`, `operation=close_completed_cpu_observed_absent` or `reserve`, the new `accounting_mode`, exact `bindings`, five exact `expected_before` pins, and fresh timezone `expires_at_utc`. Require:

```text
all_budget_writers_use_scheduler_mutex = true
all_consumers_and_launchers_use_terminal_success_guard = true
all_terminal_v1_writers_finished_before_successor_activation = true
historical_records_mutation_authorized = false
global_rebase_authorized = false
scientific_execution_authorized = false
GPU_authorized = false
```

A closure also requires the three existing cooperative revocation flags: `scope_write_authority_revoked_for_all_cooperative_project_jobs=true`, `closed_scope_future_writes_and_relaunch_revoked=true`, `external_NFS_writers_outside_observation=true`.

The exact closure fields are `key`, `parent_id`, `storage_ref`, `completion_ref`, `input_ref`, `launch_ref`, `observation_ref`. Scope is restricted to original `H038/ap_expanded_val_000` through `..._005`, original `data/expanded_full_panel_v1/PARENT`, 3GB allowances, and recorded PID 4163579 through 4163584. Input/storage/launch must be the original sibling JSON records in `research/reviews/expanded_full_panel_v1_root/PARENT`; completion is the real owned `expanded_parent_completed.json`. No completion/wait receipt is created or rewritten by the helper.

Root creates a NEW `observation_ref` receipt with exactly:

```text
kind = project_budget_CPU_completed_observed_absent_v1
root_authorizer = root_agent
status = completed_CPU_scope_observed_absent_launcher_exit_unknown
parent_id = exact parent
launch_ref = original launch ref
completion_ref = real completion ref
worker_pid = exact original launch PID
observation = {pid, absent:true, method:"targeted os.kill(pid,0) ESRCH", checked_utc:timezone timestamp}
launcher_exit_status_known = false
launcher_returncode = null
```

The actual launcher record must also have `returncode=null`. Scientific completion must declare generation/Relate complete and no GPU/outer evaluation, bind the exact original admitted source/config/registry/runtime/storage, and bind same-parent generation, observed-entry, and inference metadata. Inference must have status `post_five_iterations_inferred`, no truth/oracle read, and exactly five recorded native outer-helper returncodes zero. Those are the original synchronous helper waits. The certificate explicitly preserves `native_child_statuses_independently_asserted=false`; it does not invent a launcher exit or independent subprocess status evidence.

Apply checks only the recorded worker PID for fresh absence, measures the exact owned scope once with the unchanged stat-only conservative inventory, rechecks absence, and appends credit equal to original allowance minus retained bytes. Duplicate CPU/GPU/failure/success closure is rejected. No global process/FD scan or physical rebase occurs. All successor consumers reject closed writes/relaunch before scope writes. The original final 950GB scheduling/1TB limit remains.

## Fit-ready integration

`src/arggnn/train_mutation_routed_full_panel_pyg_v1.py` SHA `7172ac5e1b8cbb980a7df9e4c3e667d79fdc48da8c3ab9a5cea02175a656f697`.
`configs/mutation_routed_full_panel_pyg_v1.json` SHA `ae77cb496a9e8bef2e89fac3f5bb87bc627f748136aa4c34ecb2e58326964da4`.
`scripts/launch_reserved_gpu_terminal_scopes_v1.py` SHA `5b919f7b9f750f5d4f3945ee4e33a6671c64170afa59d1932b80c68bec2ba326`.

These unadmitted files now use the successor mode/helper. Wrapper pins the successor and executes the pinned original GPU launcher template, substituting only accounting labels and its locked storage check. GPU selection, canonical lease, title propagation, child handling and wait paths remain unchanged. The trainer binds exactly the twelve v4 cache sources from frozen prep `ee977bbc56f3c1b1fb71b74928ae3f065c1f7baca261fbf96fe5bdcefa3f7719`. Use config `observation_cache_source_refs` for the input's same field; use `fit_integration_receipt.json:source_sha256_for_root_input` as the exact source closure.

The fit input schema remains `mutation_routed_full_panel_pyg_input_v1`, with the existing exact metadata roles: config, registration, generation_config, runtime, storage, observation_cache, target_bindings. Root grants its original active single-arm CUDA fit capability and fresh exclusive 5GB output reservation. Storage `budget_components.consumer_adapter` must reference the successor; include the actual new wrapper, original base lease launcher, all five budget bindings, and fit/config/runtime sources in consumer bindings. Initial pins come from state after that reservation. The cache manifest is the actual completed v4 cache; no incomplete cache is consumed.

Source/cache admission is once. All 40 complete observed parents are decoded/prepared once; exact TRAIN class-span mass is aggregated on native holds once and retained. The GPU bank residency/prefetch and complete-parent backward/optimizer loop are unchanged. Each parent step uses only original expiry/lease/logical-job/RSS/CUDA guards. Storage boundaries occur for initial owned output/CUDA qualification and actual saves, raw seals and pruning, with no budget/corpus/hash replay per backward. The frozen 30-pass/960-update maximum, scheduled8-VAL selection, raw FP32 seals, equal-parent Present182 scoring, early stop, architecture/parameters and outer holdout are unchanged.

Root owns true apply, admissions, scheduling and GPU execution. Author performed metadata-only replay and focused mocked proof checks; no actual arrays, PID observations, scope measurements, cleanup, launches or commits.
