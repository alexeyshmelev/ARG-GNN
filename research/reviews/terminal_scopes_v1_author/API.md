# Terminal failed CPU preparation overlay

Stable module: `src/arggnn/project_storage_terminal_scopes_v1.py`, SHA256 `f06cf7028af7e86deb54caa1e8a01f0110db47b0d992bfe8aec8ee013ab7d79c`.
Mode: `journal_terminal_scopes_credit_v1`. New append-only journal: `.runtime/storage/project_budget_terminal_scopes_v1`.

Original V4 ledgers, V9 journal/core, CPU overlay/helper, admitted sources, historical evidence and GPU lease code remain unchanged. The frozen CPU module (`2a33a19c1447442fc99708bf3d73c21e1720a2453a99deb71d1688a4dcc5c6ff`) runs in an isolated namespace. Exactly one final ceiling predicate is changed: its upper ceiling test is deferred; its hard-limit identity and nonnegative charge checks remain. All original successful CPU/GPU proof validation still executes. This new reader enforces the combined final charge below 950GB scheduling and 1TB hard ceilings, including 64KiB per new event. Existing speculative allowances stay charged unless an individually bound closure proves a genuine terminal scope. No global rebase or physical disk measurement occurs.

## Public API

- `inspect_v1(expected_source_sha256=CORE_SHA, expected_launcher_sha256=BASE_SHA)` returns the combined charge and five pins.
- `apply_close_failed_cpu_v1(authority_ref, expected_source_sha256=CORE_SHA, expected_launcher_sha256=BASE_SHA)` performs the true failed closure. Root owns actual application and cleanup.
- `apply_reserve_v1(authority_ref, ...)` appends a fresh exclusive source-bound reservation, using all original frozen CPU reservation checks.
- Compatible consumers call `budget_check_v9`, `validate_storage_v9` and `PilotOutputV9`. Exact new SOURCE bindings are mandatory.
- `check_storage_locked_v1(storage, first=False, boundary=False, required_write_bytes=0)` is the launcher entry point while the existing canonical scheduler mutex is held. It never reacquires that mutex.

`CORE_REF` is the current exact reference to `scripts/project_budget_v9.py`; `BASE_REF` is the exact reference to `scripts/launch_reserved_gpu_v6.py`. Root authority bindings are exactly:

```json
{
  "implementation": "CORE_REF object",
  "launcher": "BASE_REF object",
  "cpu_overlay": {
    "path": "src/arggnn/project_storage_cpu_closed_scopes_v1.py",
    "sha256": "2a33a19c1447442fc99708bf3d73c21e1720a2453a99deb71d1688a4dcc5c6ff"
  },
  "terminal_overlay": "NEW_SOURCE_REF object"
}
```

Each reference above is an actual `{path,sha256}` object, not the explanatory strings. New reservations must include all four refs in `consumer_bindings`, the existing `src/arggnn/project_budget_v9_consumer.py` stub, and all actual scientific source/config/runtime dependencies. GPU jobs must additionally bind their actual new launcher and all original runtime/lease dependencies. A new wrapper may substitute this locked storage check while preserving existing GPU selection, lease, process-title and wait logic.

## Root authority

Use `kind=project_budget_terminal_overlay_root_admission_v1`, `root_authorizer=root_agent`, `active=true`, `operation=close_failed_cpu` or `reserve`, `accounting_mode=journal_terminal_scopes_credit_v1`, exact `bindings`, a fresh timezone `expires_at_utc`, and exact five-key `expected_before`:

`legacy_ledger_sha256`, `supplemental_ledger_sha256`, `journal_tail_sha256`, `cpu_overlay_tail_sha256`, `terminal_overlay_tail_sha256`.

Declare `all_budget_writers_use_scheduler_mutex=true`, `all_consumers_and_launchers_use_terminal_overlay_guard=true`, `historical_records_mutation_authorized=false`, `global_rebase_authorized=false`, `scientific_execution_authorized=false`, and `GPU_authorized=false`. A closure also requires `scope_write_authority_revoked_for_all_cooperative_project_jobs=true`, `closed_scope_future_writes_and_relaunch_revoked=true`, and `external_NFS_writers_outside_observation=true`.

For reserve, `reservation` retains the original five fields: `key`, canonical absolute `output`, `auxiliary_scopes`, integer `reserved_bytes`, and `consumer_bindings`. Returned value adds its immutable `root_admission`. Storage retains `kind=project_budget_v9_storage_record`, the new mode, and `budget_components={implementation:CORE_REF,launcher:BASE_REF,consumer_adapter:NEW_SOURCE_REF}`. `initial_ledger_pins` contains all five pins from the state after its reservation event.

## Failed CPU proof

The exact `closure` fields are `key`, `storage_ref`, `failure_ref`, `started_ref`, `input_ref`, `launch_ref`, and `drain_ref`. The evidence must bind the actual same-scope cache `failed.json` and `started.json`, original admitted storage/input, exact CPU command, known admitted producer source and all its source/config/registry/runtime pins. An existing `completed.json` is rejected. The two accepted pinned producers are `prepare_expanded_pyg_observations_v1.py` and `..._v2.py`; this is deliberately a narrow cache-preparation failure proof.

Root creates a NEW `drain_ref` receipt with these exact fields:

```text
kind: project_budget_failed_cpu_drain_receipt_v1
root_authorizer: root_agent
status: failed_CPU_preparation_workers_drained
evidence_method: pinned_finally_after_owned_ProcessPool_drain
input_ref, source_ref, failure_ref, launch_ref: exact corresponding original refs
events_ref: exact owned scope events.jsonl ref
pids: unique sorted list including launch.pid and every recorded worker_pid/prepare_worker_pid
observations: corresponding list of {pid, absent:true, method:"targeted os.kill(pid,0) ESRCH", checked_utc}
owned_process_pool_drained: true
GPU_used: false
```

Any other PIDs must be known owned children. Observations require a timezone, cannot be future-dated, and must occur after the bound drain event. Application rechecks every listed PID with targeted `os.kill(pid,0)` before the measurement and again afterward. Presence, zombies/reuse or uncertain permission fails closed. No global process/FD/mapping scan occurs.

The drain proof is the actual final `worker_cleanup_failed` event with its `scratch.rmdir()` traceback and `Errno 39` at the owned `worker_tmp` path. Both exact pinned sources execute `_drain(executor, failed)` before that operation. `_drain` uses the executor's owned process registry, joins/terminates/kills only its own workers, calls `shutdown(wait=True,cancel_futures=True)`, and checks every owned process is no longer alive. This post-drain event is thus used as drain evidence while preserving the failure result. The actual preceding failed event must exactly match `failed.json`; completed events are rejected. No successful generation/Relate/GPU receipt is synthesized.

Root must remove only owned stale sockets after its PID/ownership proof. The unchanged strict V4 stat inventory rejects sockets, links and unsafe namespaces. Regular stale temporary files and every original declared auxiliary scope remain included. Application performs exactly one actual owned-scope stat inventory, records the retained-byte charge, and credits allowance minus retained bytes once. Returned/certified status is explicitly `failed_CPU_preparation`, with `generation_completed=false` and `GPU_used=false`.

All guards reject closed keys before any scope write or relaunch. Accounting/proof metadata replay is cached with dependency fingerprints; closure scopes are never reinventoried by consumer guards. The revocation contract is cooperative and does not claim visibility into external NFS writers. Root must route all future cooperative writers/relaunchers through this reader or revoke them; frozen older guards themselves cannot know the new journal.

## Qualification and limits

`PYTHONPATH=src:. .runtime/env/bin/python -B research/reviews/terminal_scopes_v1_author/qualify.py` performs read-only current metadata replay and focused in-memory mock proof/transition checks. Actual v1/v3 failure/input/source/launch/post-drain event bindings validate. False drain, GPU, root-authorizer and PID claims, duplicate closure and final ceiling equality are rejected. A mocked transition checks exactly one measurement and exact allowance-minus-retained credit. A terminal replay fixture combines two failed certificates and a new reservation after a synthetic 954GB intermediate projection before enforcing the final ceiling. The writer methods use the new guard namespace.

Observed pre-terminal projection: `949569418037` bytes; CPU overlay 56 events; terminal events zero. No true apply, journal creation, scope inventory, PID observation, cleanup, scientific execution, GPU execution, installation, commit or push occurred. Mock root receipts were never persisted. No new GPU terminal closure operation is included; original GPU success proofs remain intact.
