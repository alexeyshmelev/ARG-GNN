# GPU4 cache-C3 / canonical H050 shared pool successor

Source only. No pool has been installed, admitted, started, or used here. No guard,
process/GPU probe, wait, signal, lease mutation, or accounting operation was called.

The staged pool is an additive successor of the reviewed cancellation fix. Its
existing exclusive physical lock, scheduler/device2/3 restriction, exact GPU UUID,
two-project-device limit, title, pending-peak memory accounting, original storage
guard, direct Popen PID/start capture, exact pidfd TERM/KILL, adopted waitpid,
and terminal/drain rules are preserved.

The new reader traverses only the controller's /proc/<PID>/task/*/children plus
exact previously recorded PID/start roots. PR_SET_CHILD_SUBREAPER is enabled
before any worker. Escaped-session descendants remain reachable through live
ancestry or adoption. New unknown adopted children, including zombies, remain
ambiguous and cannot be silently reaped or declared drained. Own metadata
permission failures or non-stabilising discovery fail closed and preserve known
owned cancellation. No unrelated PID/FD inventory or absence proof is required.
This assumes the exact trusted worker closure does not delegate lease descriptors
or scientific work to unrelated jobs. Foreign jobs are charged by NVML and are
never adopted or signalled. External sharing is explicitly authorized by the new
root capability.

The pool also preserves outer CPU/deadline supervision bypassed by direct canonical
dispatch: C3 has128GiB and H05064GiB owned RSS caps, plus each job's original
86400s monotonic deadline and the earlier pool execution expiry. Per-job sampled
RSS includes controller RSS, its assigned tree, and any ambiguous owned RSS
conservatively. A threshold breach records fail-owned-supervision without throwing out of
discovery, so every draining tick still reaches actual waitpid and Popen.poll.
Original exact-owned cancellation handles a breach; no foreign
process is sampled. H050's one runtime/spec/storage expiry variable is clipped
to the route expiry, and load_cap rejects worker authority beyond pool expiry.

Whitelist: cache-C3 v3 and canonical H050 only. TRAIN caps are 40GiB and 28GiB.
Two 2GiB per-job margins plus 4GiB headroom make 76GiB nominal, before actual
external/driver usage; this is not a fit result. The existing memory checks still
require both projected capacity and actual free memory.

The fresh route requires the separately reviewed marker-repair expiry renewal
source/bootstrap/certificate before original guards load. Its expiry must cover
route expiry +600s drain. All fresh reservations bind these refs directly.
A separate cache-C3 observer emits only successful whole-parent update counts
at1/every10 via its exact frozen trainer global binding; science stays unchanged.

Future root flow:

1. Preserve the current C3 and native GPU5 jobs. Consume the original C3's actual
   terminal direct wait and its own drain; revoke all old C3 launch/resume/write
   capabilities. Do not terminate or migrate native GPU5. Fill the proposed
   migration gate using genuine old input/launch/actual-wait references. No new
   historical PID/group absence claim is manufactured by this source.
2. After independent source review, install the exact staged pool and launcher
   to the paths in installation_manifest.v1.json. Keep the old reviewed files.
   Root must compare hashes. Both canonical shims already matched reviewed bytes
   when source was inspected; fresh runtime loading rechecks them.
3. Fill and seal a NEW active shared route from shared_route.proposed.inactive.json:
   actual GPU4 UUID, explicit expiry, exact migration gate, installed source refs.
   The existing exclusive physical lock must become available naturally. A live
   legacy exclusive job on GPU4 cannot coexist with the shared pool.
4. Run root_shared_admission_v1.py bind-h050 against the sealed route. This writes
   separate shared config/manifest/request files and preserves the existing
   exclusive canonical config/request byte-for-byte.
5. Fill the two worker-admission requests. admit-c3 reuses original C3 metadata
   validators/science/config but creates a NEW H041/C3-GNN-cachev3-shared-TRAIN-v1
   key and separate packet/output. admit-h050 reuses the frozen original H050
   admission function with only fresh shared packet/output/job namespace and
   added actual operational refs. It uses the canonical shim directly.
   Each original guard reserve directly binds actual installed pool, launcher,
   base, adapter, canonical shim, and shared route before reserve.
6. Fill controller_admission.proposed.inactive.json with both genuine admitted
   refs. assemble reserves a separate explicit 32-64MiB controller scope, creates
   the final plan and exact active capability, then validates the capability.
   Final plan is produced AFTER worker admissions to avoid hash cycles.
7. Start one controller using the new launcher; run the two declared logical jobs
   through it. A queued job still needs actual memory capacity. Pool wait results
   are genuine new pool records, not relabelled original exclusive wait receipts.
   Native GPU5 remains untouched.

All actions except describe require --execute --record PATH --sha256 SHA256.
Suggested command structure (root supplies actual immutable refs):

    PYTHONPATH=.:src:scripts .runtime/env/bin/python -B \
      research/reviews/GPU45_C3cache_H050_pool_successor_20261016_v1/root_shared_admission_v1.py \
      bind-h050 --execute --record ROUTE_PATH --sha256 ROUTE_SHA256

Use actions admit-c3, admit-h050, assemble with their corresponding sealed root
requests. After assemble, use the new launcher with commands start/run/wait/status/
cancel/shutdown and --execute --capability CAP_PATH --sha256 CAP_SHA [--job JOB_ID].

Scope is the two fresh TRAIN jobs. Future H050 EVALUATE and C3 VAL stages require
their original genuine train completion/lineage plus a separately reviewed shared
terminal/lineage integration; this packet does not relabel its pool receipts as
the original exclusive supervisor's launch/wait schemas. Ordinary original
scientific completed markers remain unchanged. No completion/storage credit is
created here, and no live lease is converted.
