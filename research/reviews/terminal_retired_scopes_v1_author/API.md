# Two-scope retirement successor

Frozen module: `src/arggnn/project_storage_terminal_retired_scopes_v1.py`, SHA256 `bbbdf26206e4e1725c79d2558b4426af848df4cbbd5ac2d0f2bc7051c0606bab`.
Mode: `journal_terminal_retired_scopes_credit_v1`. The journal is the SAME existing `.runtime/storage/project_budget_terminal_scopes_v1`.

Previous terminal v1 and terminal-success helpers, journals/events, admissions, scientific sources, active H038 trainer/config/wrapper, and successful CPU artifacts remain unchanged. The new reader validates both immutable prior prefixes, then accepts new `retire_obsolete_scope` or `reserve` events with the additional retired-helper binding. An earlier-mode event after retirement activation is rejected. Root must finish both H038 GPU wrappers and every terminal-success writer before any new event. No live retirement/admission/cleanup is performed by this author.

## Public APIs

`inspect_v1(expected_source_sha256=CORE_SHA, expected_launcher_sha256=BASE_SHA)` returns the combined charge.
`apply_retire_obsolete_scope_v1(authority_ref, expected_source_sha256=CORE_SHA, expected_launcher_sha256=BASE_SHA)` performs one true typed retirement.
`apply_reserve_v1`, `budget_check_v9`, `validate_storage_v9`, `PilotOutputV9`, and `check_storage_locked_v1` retain their existing compatible signatures. The locked check requires the existing canonical scheduler mutex held and does not reacquire it. New guards reject retired keys before writes/relaunch; all retired scopes remain occupied and cannot be reused by overlapping reservations.

New root bindings are the previous five `implementation`, `launcher`, `cpu_overlay`, `terminal_overlay`, `terminal_success_overlay`, plus `terminal_retired_overlay` referencing the new source. All six refs must appear in new reservation consumer bindings, along with the original v9 consumer stub and actual scientific source/config/runtime/launcher refs. All old component refs keep their prior frozen hashes. The five initial/expected pins remain unchanged: `legacy_ledger_sha256`, `supplemental_ledger_sha256`, `journal_tail_sha256`, `cpu_overlay_tail_sha256`, `terminal_overlay_tail_sha256` (the SAME shared chain).

Root authority has `kind=project_budget_terminal_retired_root_admission_v1`, `root_authorizer=root_agent`, `active=true`, `operation=retire_obsolete_scope` or `reserve`, the new `accounting_mode`, exact `bindings`, fresh five-key `expected_before`, and a fresh timezone `expires_at_utc`. Require:

```text
all_budget_writers_use_scheduler_mutex = true
all_consumers_and_launchers_use_terminal_retired_guard = true
all_terminal_success_writers_finished_before_retirement_activation = true
historical_records_mutation_authorized = false
global_rebase_authorized = false
scientific_execution_authorized = false
GPU_authorized = false
```

Retirement additionally requires:

```text
scope_write_authority_revoked_for_all_cooperative_project_jobs = true
closed_scope_future_writes_and_relaunch_revoked = true
all_original_and_resume_capabilities_for_retired_scope_revoked = true
external_NFS_writers_outside_observation = true
```

Use authority field `retirement` with exact fields `key`, `retirement_kind`, `evidence_refs`, `observation_ref`. The module exports `CASES`: exact allowed keys, retirement kinds, allowances, output paths, PID lists, and role-to-path evidence maps. Construct actual refs as `{name: reference(ROOT,path) for name,path in CASES[key]['evidence'].items()}`. Every original reservation/root/source/config pin and each supplied evidence hash is verified. Source roles also have hardcoded original proof hashes.

## Exact proof cases

1. `budgeted_shared_pool_v3/gpu3_fits_v1/mechanisms303-cache-v1`, kind `failed_before_output`, allowance 52,321,048,576 bytes. The exact absent output is `results/biological_mechanism_controls_v1_seed303_cached_gpu3_v1`; auxiliary scopes are empty. Bind original pool admission, closed registry, complete owned failure log, actual cache-admission/storage-snapshot refs, supervisor launch/source/exit, controller start, paired shutdown, and exact controls/pool source. Worker exit is known 1; the supervisor exit status is not independently known. The bound error/source prove failure at storage admission before output creation. The helper does not call it never-launched or completed, and does not infer GPU-use status from absence. The separate 100MB controller/log reservation stays charged. Required recorded PID set is `[1313616,1313627,1313663,1313689,1313732]`.
2. `anchor-parent-full-support-CPU-v2-r001`, kind `abandoned_partial_CPU_matrix`, allowance 40,000,000,000 bytes. The exact retained output is `experiments/results/anchor_parent_full_support_CPU_profile_v2_r001`; auxiliary scopes are empty. Bind original matrix launch/source/plan/start/failure, first-cell launch/admission/actual zero wait, successful evidence, root recovery/terminal, and unlaunched resume plan/source. The original matrix remains failed/abandoned; the successful `age_reset_float64_cp0` cell remains successful. Both actual 1,072,691,832-byte artifacts must remain regular, singly linked, with their observed inode/stat identity preserved. Only targeted identity stats are read; no payload hashes, Torch decoding or numerical reevaluation occurs. Resume execution markers must remain absent. Original active matrix/resume capabilities require explicit NEW revocation. Required recorded PIDs are `[1386393,1386398]`.

Root creates a NEW observation receipt with exactly:

```text
kind = project_budget_obsolete_scope_terminal_observation_v1
root_authorizer = root_agent
key = exact allowed key
retirement_kind = exact CASES kind
evidence_refs = exact request evidence map
pids = exact CASES sorted PID list
observations = corresponding [{pid, absent:true, method:"targeted os.kill(pid,0) ESRCH", checked_utc:timezone timestamp}]
all_original_and_resume_capabilities_revoked = true
```

Observations cannot be future-dated. Apply fresh-checks only these recorded PIDs for ESRCH, performs exactly one unchanged conservative owned-scope stat inventory, rechecks PIDs, and appends. Present/reused/zombie/permission-uncertain PIDs fail closed. A truly absent mechanism scope must measure zero entries/bytes. Anchor retained artifacts and metadata stay charged. Credit is original allowance minus actual retained bytes, once, plus 64KiB event metadata charge. No files are deleted or rewritten. Certificates explicitly set `scientific_completion_claimed=false`; the anchor records its preserved successful cell and actual worker exit 0. No launcher zero or completed matrix is synthesized. Original final 950GB/1TB ceilings remain.

## Future native admission

Future native driver sources are unadmitted and may pin this helper/mode before admission. Storage remains `project_budget_v9_storage_record`, with `budget_components={implementation:CORE_REF,launcher:BASE_REF,consumer_adapter:RETIRED_REF}`, active guard declaration, exact own reservation/output/allowance, and five initial pins from the state after reserve. Include all six accounting refs and actual native guard source closure. H038 and any other currently admitted sources must remain on their frozen guards until terminal; appending a new retirement event before that barrier would cause them to fail closed.

`qualify.py` replays the real prior metadata and uses only in-memory MOCK root observations, measurements and events for focused typed proof/replay/one-inventory/duplicate/fabricated-completion/barrier checks. It performs no actual PID observation, owned inventory, journal append, artifact payload read, cleanup, scientific execution, installation or commit. See `receipt.json` and `inspection.md` for the handoff evidence.
