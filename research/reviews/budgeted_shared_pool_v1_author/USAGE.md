# Future shared pools with supplemental storage

This source-only companion creates new pools through the original audited
gpu_pool_v1.Controller. It does not modify or adopt an existing controller,
worker, socket, lease, scientific module, pinned cache source, or budget ledger.

The default CLI reads only the companion source/config. An operational pool
requires a fresh exact independent PASS for the source/config and complete
plan, an active time-bounded root admission, and reservations that root has
already registered through project_budget_v4. Root remains the only scheduler.
A pending template is not a review or execution authority.

## Composition and invariants

The companion overrides only _validate_spec and _lease_check. The original
start, acquire_device, inherited lease descriptor, device inventory admission,
two-physical-device ceiling, memory reservation, process scan, subprocess argv
and environment, FIFO, descendant draining, log bounds, shutdown and pidfd
signalling remain the original implementations. The constructor's existing
modules parameter contains only modules explicitly registered in the reviewed
plan. The finite policy includes the six original modules and exactly
arggnn.biological_bank_cache_v1.

Each registered job binds its complete normalized original spec, literal argv
hash, exact module source, worker config/capability references, unique fresh
output, exact stored supplemental reservation, memory components, and seed/model
provenance. The server rejects raw protocol submissions that differ from that
registration. The CLI accepts only a registered job name. Every lease check,
including immediately before the original Popen, revalidates source/config,
root admission, independent review, all own reservation bytes and the combined
legacy+supplemental accounting. Unrelated valid reservations can change after
the first exact ledger check. Original errors propagate or remain in the
original controller safety_error.

Storage is a read-only duck-type. No legacy controller/job reservation is
created and no supplemental mutation/rebase/reclaim occurs. One unchanged
project_budget_v4.check_reservation code object validates the complete current
ledger per guard. A same-mutex reread is bound to its returned ledger hash, and
all own registered reservations must match exactly. The external reserve plus
all legacy and supplemental reservations must also fit current filesystem
free bytes. The unchanged budget policy enforces the 950 GB scheduling ceiling
and 1 TB hard limit as decimal bytes.

Original acquire_device holds scheduler.lock while calling storage.controller.
Opening another flock inside check_reservation would self-deadlock. The
companion constructs a private function with the unchanged checker code and a
private mutex binding. It scans only /proc/self/fd and Linux fdinfo, proves a
current-process FLOCK ADVISORY WRITE on the exact canonical scheduler inode,
duplicates that descriptor, and validates identity/ownership before and after
the checker. It never acquires or unlocks that borrowed descriptor, inspects the
call stack, changes shared module globals, or patches the original sources.
Closing the duplicate preserves the original open-file-description lock.
Ordinary boundaries enter the original budget mutex, then use the same proven
owned-descriptor composition. Current free-space checking and all-own ledger
comparison occur while that mutex remains held. Missing ownership proof fails.

A new socket is inside its controller's reserved directory. The controller
reservation must cover at least MIN_CONTROLLER_RESERVE (64 MiB) plus every
registered bounded log. The original 32 MiB atomic registry cap bounds its
concurrent registry+temporary-file footprint. Supplemental stat measurement
does not accept Unix sockets, so the controller callback checks its reservation
and accounting without scanning that controller scope. Job output/auxiliary
scopes use the unchanged stat boundary check at submission. Original bounded
logs/registry constrain controller writes. These are contractual storage
reservations, not an OS quota or global remeasurement.

For cache jobs, the cached_bank component must cover the entire configured
32,000,000,000-byte GPU LRU cap rounded up to MiB. Positive additional components
are required for uncached banks/retained references, autograd, model/optimizer,
and runtime/allocator. Their sum is the requested total peak. Device headroom
is additional to that sum. Root must choose evidence-based complete bounds;
the cache cap alone is never the total worker peak. This scheduler does not
grant the cache worker its separate root profile/fit capability and does not
change any scientific configuration or claim a performance improvement.

## Root preparation for a future pool

1. Retain current workers and guards until their normal drain. Choose a fresh
   pool name and the same eligible device only when original acquisition can
   admit it. Up to two new controllers can hold distinct physical GPUs 2-5.
   Keep useful fitting workers in each controller's queue. FIFO and complete
   memory reservations determine concurrency within each device.
2. Use the independent source review and the already audited budgetv4 workflow.
   Pre-register a controller directory .runtime/gpup/POOL and every exact job
   output plus any native runtime/checkpoint auxiliary directories. Names must
   match budgeted_shared_pool_v1/POOL/controller and
   budgeted_shared_pool_v1/POOL/JOB. Include companion source/config and each
   worker source/config/capability in reservation consumer_bindings.
   Controller auxiliary_scopes must be empty. All registered controller/job
   scopes must be disjoint and fresh. Job storage_bytes equals its stored
   reserved_bytes. Do not reserve through the original StorageLedger.
3. Copy exact stored reservations, including their budget root_admission refs,
   from the root budget reserve receipt into the companion plan. Bind the
   unchanged historical legacy snapshot. Runtime bindings are explicit source,
   shell, or JSON refs, such as scripts/project_env.sh and pinned runtime guards.
   Capture exact legacy and supplemental ledger SHA256 pins after reservation.
   The author has not queried these live artifacts.
4. Complete root_admission.template.json with active=false initially. Every job
   spec includes policy and log_limit_bytes explicitly. Compute argv_sha256 as
   SHA256 of budget.encoded([python, "-m", module, *args]). Compute
   start_recipe_sha256 from the canonical start argv below with the literal
   <ROOT_ADMISSION_SHA256> placeholder, which avoids a self-referential hash.
   The actual start command replaces that token with the final authority file
   SHA256, and its real hash is returned by --validate-admission.
5. Obtain a fresh independent review for the exact source/config and
   digest(plan). The review must bind the artifact refs, use the documented
   kind/role, record actual_live_lock_or_gpu_execution_performed=false, and
   explicitly pass. The independent source review may include findings.
   Root then binds the review ref, sets active=true and a short explicit UTC
   validity window, and computes the final root authority hash.
6. Run --validate-admission. This validates metadata, sources and stored
   reservation authorities only. It does not access live ledgers, device
   inventory, scope contents, locks, sockets or scientific modules.
7. Root alone may start the fresh pool and submit only its registered jobs.
   Startup performs exact first-ledger-pin validation, then original device
   acquisition. It refuses an occupied device, a third physical device, or an
   existing pool directory. Nothing is attached to a running pool.
   Submit each reviewed job to its intended device controller. Use --status
   and a natural --shutdown when all desired work is done.

The plan and job list are sealed for that pool. Adding unregistered jobs
requires a new reviewed future pool after natural drain; this version does not
hot-add authority or change a running controller.

## Exact commands

All commands execute on anogena inside the project environment. Replace the
capitalized authority/job fields only with final independently reviewed root
artifacts. Source/config hashes are fixed in author_receipt.json.

Source-only report:
~~~sh
ssh -o BatchMode=yes anogena 'cd /workspace-SR003.nfs2/ARG-GNN && source scripts/project_env.sh && PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python -m arggnn.budgeted_shared_pool_v1'
~~~

Bounded CPU/fake checks:
~~~sh
ssh -o BatchMode=yes anogena 'cd /workspace-SR003.nfs2/ARG-GNN && source scripts/project_env.sh && PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python research/reviews/budgeted_shared_pool_v1_author/check_bounded_fake.py'
~~~

Metadata-only authority validation:
~~~sh
.runtime/env/bin/python -m arggnn.budgeted_shared_pool_v1 --validate-admission --admission ROOT_ADMISSION_PATH --expected-admission-sha256 ROOT_ADMISSION_SHA256 --expected-source-sha256 SOURCE_SHA256 --expected-config-sha256 CONFIG_SHA256
~~~

Canonical future start recipe:
~~~sh
.runtime/env/bin/python -m arggnn.budgeted_shared_pool_v1 --start --admission ROOT_ADMISSION_PATH --expected-admission-sha256 ROOT_ADMISSION_SHA256 --expected-source-sha256 SOURCE_SHA256 --expected-config-sha256 CONFIG_SHA256
~~~

Future submit/status/natural shutdown:
~~~sh
.runtime/env/bin/python -m arggnn.budgeted_shared_pool_v1 --submit-job REGISTERED_JOB --admission ROOT_ADMISSION_PATH --expected-admission-sha256 ROOT_ADMISSION_SHA256 --expected-source-sha256 SOURCE_SHA256 --expected-config-sha256 CONFIG_SHA256
.runtime/env/bin/python -m arggnn.budgeted_shared_pool_v1 --status --admission ROOT_ADMISSION_PATH --expected-admission-sha256 ROOT_ADMISSION_SHA256 --expected-source-sha256 SOURCE_SHA256 --expected-config-sha256 CONFIG_SHA256
.runtime/env/bin/python -m arggnn.budgeted_shared_pool_v1 --shutdown --admission ROOT_ADMISSION_PATH --expected-admission-sha256 ROOT_ADMISSION_SHA256 --expected-source-sha256 SOURCE_SHA256 --expected-config-sha256 CONFIG_SHA256
~~~

The author performed 23 bounded checks on three handmade fake registered pools
and three fake jobs per pool, with only three inert subprocess objects. The
fixture exercises the original Controller methods and original acquire_device
under contained fake-root locks/sockets. It disables real subreaper changes,
injects a handmade inventory/proc reader, and replaces Popen with inert pipe
objects. A private budget instance uses unchanged code with contained fake
metadata constants. Fake PASS records are explicitly fixture-only and never
become real review/authority. Fixture cleanup removes only its fresh owned root.
The deadline is 25 seconds. No massive scaling fixture or actual GPU task ran.
