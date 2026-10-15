# Resource guard hot-path source audit

Use transaction-local I/O memoization first. It reduces duplicate NFS reads inside one original scheduler-mutex check while preserving a fresh complete accounting check on the next operation. Do not add a cross-operation state cache or throttle strict write checks.

## Exact costs

credit-v3 check_storage_locked_v1:375 and _state_locked:255 each validate the policy, giving at least 20 file-byte reads. _bindings adds another source reference. _state_locked:265–300 replays all new events and proofs without an overlay cache. Each reservation path goes through v9.successor_paths→v4.scope_spec:192–195, rehashing every consumer binding. The final own-binding loop at credit-v3:386 repeats it. v4.canonical:45 resolves paths before file reads, multiplying NFS metadata work.

The capacity-v2/v1 base already has a cache. Warm hits still fingerprint all discovered dependencies using canonical/lstat (capacity-v1:229–232, CPU overlay:61–70). It is inaccurate to say the base always fully replays. The overlay always replays.

H040 context:49–51 adds full accounting to every lightweight guard. H040 evaluation has 96 prediction guards and each RAMMatrix.seal writes a compact JSON via PilotOutputV9.save:429, which performs another strict boundary. These two sites alone cause at least 192 full transactions. Context-only throttling leaves all 96 strict seal checks. Full-C3 supervisor:309–311 polls the same full checker every 15 seconds inside its log/work-monitor loop. Source costs match root's reported contention, but this task performed no live profiling.

## Implemented source-only correction

transaction_memo_adapter_v1.py explicitly installs in a future consumer process. It wraps the original mutex context without replacing its flock implementation. Thread-local state starts after that lock is acquired and is destroyed before unlock. Nested entry rejects before attempting another flock.

It memoizes only v4.file_bytes and canonical calls from the exact pinned v4.file_bytes/reference code objects. Canonical calls from dependency fingerprints, journal enumeration, scope inventories and other callers stay uncached. Original lstat/fstat, journal listing/fingerprints, expiry/assert_open, replay, prefix validation, lease/title/RSS/CUDA guards and measure_scopes are not memoized. Before successful exit, cached namespaces and file identities are rechecked uncached. Mutation during a check therefore fails before the guard returns authority to its caller's next write.

The cache has a 32 MiB retained payload/key charge cap, 4,096 total entries and bounded key lengths. Capacity overflow delegates to the original reader. These bounds concern retained cache data, not a measured Python RSS promise. Every fresh operation rereads evidence. Original source SHA and function/code identities are recorded. Exceptions clear state; explicit uninstall restores exact original function objects.

Future integration: bind this helper in the new consumer's admitted source closure, verify the original project_budget_v4 source pin, then explicitly call install(storage_guard.budget.v4, expected_source_path=..., expected_source_sha256=...). Continue calling the original budget_check/validate/writer APIs with their original arguments. Keep the returned handle for receipt and explicit lifecycle cleanup. Install before consumer threads start. This adapter is for consumer read/check transactions, not root accounting mutations such as reserve/close/apply. It does not authorize any reservation or launch, change accounting mode/policy pins, or modify an active job.

## Qualification and limits

13 handmade fixture cases passed using AST-extracted original v4 I/O helpers and fixture-only lock/journal files. Repeated reads fell from 24 to 2 within one operation, and the next operation performed 2 fresh reads. The cases cover matching acceptance/measurements, SHA/JSON/path/revocation rejection, between-operation tamper/revocation, tamper during a transaction, fresh scope measurement, bounded fallback, nested/exception cleanup and thread isolation. No live guard was imported/configured and no live journal or scientific payload was read. The initial harness assertion failure and fixtures remain preserved in qualification_attempt1.json.

This is a plausible substantial reduction when reservations share source bindings. It leaves journal parsing, overlap checks, dependency fingerprints and unique reads intact, so fixture ratios are not a production speedup estimate. Keep every prediction/resource guard and strict planned-write boundary initially. If root later measures remaining overhead, evaluate further changes separately. Do not use time-only throttling to claim immediate journal revocation.
