# GPU pool v1

This controller holds one genuine canonical GPU flock and passes that identical
open-file description to several scientific Python workers. It never creates a
CUDA context. The root scheduling agent owns production activation, experiment
registration and review.

## Lease and physical-device admission

Start uses the existing .runtime/locks/scheduler.lock and the exact
gpu_2.lock through gpu_5.lock files used by scripts/launch_gpu.py. Under the
scheduler lock it counts held eligible device locks, rejects a third distinct
device, and acquires only the explicitly requested free device. It checks real
driver inventory for zero compute PIDs, zero utilization and at most 100 MiB
observed memory. GPUs 0, 1 and 6 are never eligible.

The lease contains the existing six-field launcher schema: launcher_pid, gpu,
uuid, job, started_unix and storage_bytes. Its job is stable:
gpu_pool_v1:<pool>:gpu<physical>. Metadata is written once before workers start
and never rewritten while the lease is held. Workers inherit the controller's
actual FD via Popen(pass_fds=(lease.fileno(),)), without a shell. The controller
never explicitly unlocks that shared FD. Closing it after a controller failure
does not release the kernel lock while an inherited worker descriptor survives.

ARGGNN_JOB_NAME remains the stable physical pool job for unchanged guards.
ARGGNN_LOGICAL_JOB_NAME is the distinct requested worker job. Registry records
preserve requested model, seed, literal argv, source SHA256, output path, PID,
start time, descendant identities, memory/storage admission and exit status.
The pool does not rewrite worker seeds, configurations or scientific arguments.

Production admits these GPU entry modules:

- arggnn.biological_candidate
- arggnn.biological_gst_control_v1
- arggnn.biological_mechanism_controls_v1
- arggnn.official_argmix_biological
- arggnn.train_official_argformer_biological
- arggnn.native_argformer_exec

The scientific workers retain their pinned lease, exact
AI Auto Research Pipeline Test title and actual own-PID NVTOP validation.
The controller does not manufacture title/NVTOP evidence. ARGMix and native
ARGformer retain their inherited lease FD across native exec. The common guard
uses pread, so concurrent guards do not race the shared FD offset.

GST imports only standard library at module scope. Its pinned trainer script
applies lease/title before importing scientific code. GST train reuses the
unchanged candidate configure_fit and check_fit_lease, so actual own-PID NVTOP
validation precedes model/device allocation through the same guard contract.

Mechanism controls run their module bootstrap lease/title guard before Torch
imports for CUDA training, and their trainer script repeats this guard before
control/scientific imports. They reuse the unchanged candidate configure_fit
and check_fit_lease for actual device/own-PID NVTOP qualification. Their pinned
scientific sources, admission protocol and configurations are unchanged.

## Memory accounting

Each submit requires a positive peak_mib and each pool requires device
headroom. The units are MiB, with 1024 MiB per GiB. All launched jobs' peak
reservations remain active before CUDA initializes and throughout descendant
drain. For each new admission the controller queries driver memory and compute
PIDs. Its conservative budget is:

active peak reservations + requested peak + unattributed observed memory
+ device headroom <= physical device memory.

Unattributed memory is at least the initially observed free-device overhead,
and otherwise the positive difference between device memory and recognized
worker compute memory. Any recognized job's combined compute usage above its
peak blocks further launches. Any unrecognized compute PID blocks launches,
including inaccessible PIDs. Observed memory plus headroom must also fit.
Process titles alone never establish ownership.

For an 80 GiB device, two 12 GiB candidate reservations and one 40 GiB ARGMix
reservation consume 64 GiB. With 8 GiB headroom, that leaves approximately
8 GiB less observed unattributed overhead. A fourth 12 GiB reservation queues.

policy=queue keeps unsafe work in strict FIFO order. policy=reject rejects
unsafe admission or a submission behind an existing queue. A job whose peak
can never fit the initial device budget is rejected immediately. Memory
overrun, foreign activity, query failure or lease mutation do not kill another
user's processes. They block additional work and appear in status.

## Storage admission

Start requires the SHA256 of a complete
biological_candidate_storage_snapshot_v1 receipt. It must name this exact root,
prove a successful du -s -B1 of the entire root, have nonnegative integer
used_bytes, and contain a timezone-aware completed timestamp in the past.
The controller reads this receipt and never runs recursive root du.

The scheduler lock also protects a persistent shared ledger at
.runtime/gpup/storage_reservations.json. Its budget includes:

- Complete root snapshot baseline.
- Explicit external_storage_reserve_bytes covering every unaccounted writer,
  including existing exclusive GPU jobs and later writes since that snapshot.
- At least 64 MiB controller reserve for each pool.
- Explicit storage_bytes for every accepted job, including its bounded log.

The projected total must stay strictly below the chosen ceiling, at most
450 GB. statvfs checks filesystem free-space headroom without scanning NFS.
All pools must use the same snapshot hash, ceiling and external reserve as the
existing ledger. The controller never silently rebases that ledger. A new
snapshot/rebase requires an offline root audit that accounts for persistent
outputs and outstanding writers.

Completed, failed and cancelled jobs retain their storage reserves because
outputs or logs remain. Output paths remain globally reserved in this ledger.
Pool names and logical names are fresh and cannot be adopted/reused. This is
deliberately conservative. The ledger does not measure arbitrary worker output
growth: root must give valid peak write reservations and admitted scientific
workers must enforce their own accounted output budgets. Unaccounted external
writers invalidate the snapshot contract.

The pool creates its own registry/log directories and never precreates the
scientific output directory. Each output must be a fresh canonical in-project
path with an existing parent. Worker logs are pipe-drained and capped at
log_limit_bytes; excess bytes are discarded with an explicit dropped-byte count.
log_limit_bytes must fit within the job storage reserve.

Pinned ARGMix and native ARGformer bootstraps still perform their own complete
root storage scan at startup. They are unchanged. Removing those worker scans
would require a separate scientific-source review. Candidate workers use their
existing hashed snapshot and accounted write guard.

## Lifecycle and operating-system boundary

Linux child subreaping is required and verified before admission. Orphaned
descendants, including a child that calls setsid, stay attached to the
controller. UID and process start ticks bind known identities. After process
enumeration the controller reads every kernel task child list, merges direct
children and newly subreaped orphans, and requires the child lists to stabilise
within a bounded number of observations, including a child-list recheck after
all merged process identities are read. A child-to-grandchild transition during
that final identity read forces another observation. Confirmed leader exit triggers a fresh
process and compute observation before finalization. This closes the race where
a child or grandchild appears while its parent becomes a zombie during /proc
enumeration. Unassigned controller descendants conservatively block admission
and drain, including descendants that closed the lease FD. Ambiguous processes
are never signalled just for this relationship. A leader's exit
does not free its memory reservation while its descendants, ambiguous lease
holders or compute activity remain.

Session ownership requires the original leader UID/start identity or a still
verified live member of that original session. If the numeric leader PID is
reused with a different UID/start time, the old session shortcut is disabled.
Unknown descendants remain conservative blockers and are not adopted merely
from PGID/SID numbers. Parent ancestry also requires the current parent UID.

Known/launched descendants must be inspectable. Failure to inspect a known
descendant stops admission and drain. The server may deny /proc FD access to
unrelated same-UID CPU processes. Those PIDs are reported as
uninspected_unrelated_cpu_pids; they are never adopted or signalled. Accessible
unknown processes with this lease inode open conservatively block admission and
drain, even if the file was opened independently.

This protects against accidental resource conflict and escaping workers. It
does not treat malicious same-UID code transferring inherited FDs to unrelated
processes as a supported workload. Foreign/unrecognized CUDA PIDs always block
admission. Scientific workers must retain the inherited descriptor through
their complete CUDA lifetime.

shutdown stops new admissions, cancels queued work and naturally drains running
workers. shutdown --terminate additionally sends SIGTERM only to proven own
UID/start-time identities using Linux pidfd, never a whole unverified process
group or a foreign PID. If the fixed Python lacks optional pidfd APIs, a
standard-library ctypes wrapper invokes the real pidfd syscalls on verified
Linux x86_64/aarch64. Kernel rejection fails closed, with no kill/killpg fallback.
It does not escalate to SIGKILL. Ambiguous holders may
require root investigation and remain untouched.

Only fully drained normal shutdown closes the controller's lease and removes
its own socket. Metadata/registry/logs remain for audit. Controller loss leaves
inherited leases held, and a replacement pool must not adopt a held lease or
rewrite its metadata. Status can report a stale disk registry when the socket
is unavailable. wait refuses to infer unfinished-job completion from a dead
controller.

The socket is contained at .runtime/gpup/<pool>.sock, has mode 0600, and checks
Linux SO_PEERCRED for the controller UID. Requests are bounded to 64 KiB,
one JSON request per connection. Slow or waiting clients do not pause worker
supervision. There is no production fake-inventory, test-root, alternate
interpreter or arbitrary-command CLI switch. Constructor injection exists for
contained tests.

## Root CLI

Source scripts/project_env.sh and invoke the explicit project interpreter.
These are interface examples, not authorization to activate an experiment:

~~~sh
"$ARGGNN_PYTHON" scripts/gpu_pool_v1.py start \
  --pool biological_gpu2_v1 --gpu 2 --headroom-mib 8192 \
  --storage-snapshot <complete-root-snapshot.json> \
  --storage-snapshot-sha256 <sha256> \
  --external-storage-reserve-bytes <all-other-accounted-writes>
~~~

Start runs in the foreground as a supervisor. The root agent may use its
existing project-contained process supervisor to run it persistently.

~~~sh
"$ARGGNN_PYTHON" scripts/gpu_pool_v1.py submit \
  --pool biological_gpu2_v1 --job candidate_seed202 \
  --model candidate_age_d2 --seed 202 \
  --output <fresh-output-path> --peak-mib 12288 \
  --storage-bytes <reviewed-peak-write-reservation> \
  arggnn.biological_candidate <exact-reviewed-worker-arguments>

"$ARGGNN_PYTHON" scripts/gpu_pool_v1.py status --pool biological_gpu2_v1
"$ARGGNN_PYTHON" scripts/gpu_pool_v1.py wait \
  --pool biological_gpu2_v1 --job candidate_seed202
"$ARGGNN_PYTHON" scripts/gpu_pool_v1.py shutdown --pool biological_gpu2_v1
~~~

wait polls status in the client and returns zero only for succeeded requested
work. A timeout leaves workers supervised. shutdown returns the drain state;
use wait/status to observe completion. Each submission must carry the exact
reviewed worker arguments, including the worker's own output/config/snapshot
bindings. Requested model/seed are preserved provenance, not proof that
arbitrary arguments execute that model or seed.

## Source-only validation

tests/test_gpu_pool_v1.py uses fake GPU inventory, isolated leases under the
source-audit directory, CPU helper modules and a contained instance of the
unchanged lease guard. It tests concurrent device admission, exclusive-launcher
interop, shared FD offset identity, immutable metadata, memory before CUDA
allocation, foreign PIDs/overruns, FIFO queueing, descendants, controller loss,
bounded logs, storage and socket supervision. No real driver query, reservation,
scientific launch, dataset read, installation or Git operation is performed by
this development validation. Production activation and actual per-worker NVTOP
qualification remain separate root-owned actions.
