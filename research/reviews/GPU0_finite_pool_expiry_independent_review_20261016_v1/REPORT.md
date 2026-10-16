# GPU0 finite pool expiry — independent source review

**Verdict: PASS for the sealed narrow v2 source change; no open actionable finding.** This permits root to review/admit the new route after genuine old-controller completion/drain. It grants no activation or job authority, and this review did not inspect live controller/device state.

Reviewed v2 handoff: `research/reviews/GPU0_declared_experiments_pool_source_20261016_v2/source_handoff.v1.json`, SHA256 `a05d9359095903fe4dc19368a3f50990a78028718a7f32e2d978201582c47b20`.
Final pool: `gpu0_declared_pool_v1.py`, SHA256 `0d3cb8a055a41edc33b0be99c5df9b3dcc4d1c6956d86610a286adde7e9c6f7f`.
Authoritative actual diff: `minimal_actual_diff.v2.txt`, SHA256 `a19e29ab98e99658ee9e56d8c6fb4c16e26a467658932344bc71aa6777bc82f9`. Earlier diff/validation revisions are historical.

## Confirmed v1 defect

V1 `finite_jobs_v1.py:204–207` requires controller expiry no later than every method/envelope expiry. TGB `common_v1.py:78–81` restricts its unchanged input to at most7200seconds. Including TGB therefore limits the whole controller to that short expiry. The inherited `_owned_resource_check` and `tick` then invoke owned abort/drain at global expiry, even if native authority remains valid. This is an actual source incompatibility for mixed durations, rather than an observed runtime failure.

## Minimal v2 correction and preserved boundaries

I independently regenerated the complete diffs of all five facade files; they exactly match the authoritative patch. Apart from new route/hash bindings and reuse of the original sealed catalogue, behavior changes are confined to the expiry relation, one metadata helper, two launch gates and one active-loop clause.

- The finite map now requires `envelope_expiry <= unchanged_method_expiry <= controller_expiry` (v2 contract lines215–217). It does not require every immutable map entry to remain current on later client/capability reads. A completed or never-started expired short entry therefore does not invalidate continuing native work.
- `job_expiry_v2` authenticates the exact operational envelope, reconstructed outer job, storage identity and original method input, and returns the envelope expiry only after proving it does not extend the original method expiry. It neither edits nor renews those authorities.
- An already-expired queued row is cancelled before slow scan/memory/guard/log/spawn work, then other queued work can be considered. A second check authenticates expiry metadata **before** sampling the clock immediately at the original Popen boundary. Expiry during preparation follows the original no-child failure path, with no scientific process started.
- Each ACTIVE owned row is additionally checked against its original absolute expiry in the inherited RSS/deadline loop. A violation feeds the original non-throwing failure/drain mechanism; repeated ticks still reach termination service and actual waits. Original global pool expiry, monotonic deadlines, RSS limits, TERM/KILL authority and finite drain window remain. This failure policy is still pool-wide: an ACTIVE job that expires can abort the owned pool. A short job that has genuinely completed/drained leaves ACTIVE and no longer truncates a valid longer job.
- The unchanged dispatcher still requires current envelope/original input authority before invoking the exact original runtime. Every original scientific runtime/guard, cap, source closure, input expiry and deadline remains in force. No three-way overlap or new scientific input is introduced.

The catalogue,64-direct-ref ceiling, distinct small operational charges,32–64MiB controller checker scope, GPU0 UUID/exclusive lifetime FD, memory accounting/headroom, ownership, log, wait and drain bodies remain inherited. No new framework or arbitrary module route was added.

The retained native input/storage/caps pins and its7GiB scientific charge remain exact. Its absolute expiry is still `2026-10-18T23:41:32.476768+00:00`; future launch does not restart a72hour clock.

## Validation and root boundary

All v1 readiness references, final v2 handoff references and original retained native metadata pins matched their hashes. I parsed/compiled the five facade sources and statically reconstructed/compiled the transformed queue/owned-resource bodies without executing them. The final gate's operand order, immediate Popen placement, ACTIVE guard and preserved global-expiry/failure clauses were checked on those actual transformed source bodies.

No source module, metadata stub, Torch, fixture, guard, admission, job, host/device probe, payload or Git operation ran in this independent review. Only this new review folder was written.

Root still needs genuine old-controller terminal0/closed registry/owned drain/permanent write revocation, genuine never-started native reuse evidence, original ready phase inputs, fresh source-bound operational/controller admissions and real memory/lease checks. Maximum concurrency2 is sufficient for the proposed pair sequence; memory fit and actual completion remain unmeasured. This source fix supplies compatible controller duration without extending any scientific authority.
