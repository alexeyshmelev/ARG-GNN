# Independent GPU pool v1 review

Verdict: PASS for the final source revision and contained CPU tests.
Certification applies only to controller SHA256
ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1.
Earlier pool revisions are not certified. Root owns production activation and
all actual per-worker CUDA/NVTOP qualification.

## Evidence

The final author suite passed25 tests in the independent replay.
Eight additional independent checks passed, including the precise
descendant-to-grandchild transition during the final identity merge and
reused-leader PID rejection. Source pins were verified before and after
these tests. The author receipt for this revision is
research/reviews/gpu_pool_v1_source_audit/receipt.json at SHA256
5c57573c8484aa2f21747cdc38acf02454377aa5dc6a5c0224166827433be6ec.

| Final artifact | SHA256 |
| --- | --- |
| src/arggnn/gpu_pool_v1.py | ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1 |
| scripts/gpu_pool_v1.py | 9bb91725f09cc451e80734b03b102c9f624b9df8a7fdfe3ec911330bc8ebdb8f |
| tests/test_gpu_pool_v1.py | d6347d1cefbf70f17db116f5a6ca40bfa4ce3c604551b35d211d98c17795c73f |
| docs/gpu_pool_v1.md | c8b15bdbfbc84365c830e57cd3c40c47ee1dea01554659c2bb15314941e48ae5 |

## Physical lease and memory contract

The controller shares the existing scheduler lock and canonical physical
gpu_2 through gpu_5 locks with the unchanged legacy launcher. Acquisition is
atomic under the scheduler lock and counts held distinct physical devices.
A third held device is rejected. Duplicate physical indexes or canonical
UUIDs, excluded devices, incomplete inventory, existing compute activity,
nonzero utilization, or excessive idle memory cannot admit a controller.

Multiple workers receive the identical inherited open-file description
through pass_fds. This is stronger than equal path or inode. The contained
tests changed the controller FD offset and observed that same offset in
both independent workers. The guard uses pread, so its concurrent reads
do not disturb that shared offset. The pool writes the six-field lease
metadata once before workers exist and never rewrites it during the lease.
Closing the controller's descriptor does not explicitly unlock a surviving
worker's inherited descriptor. Controller loss cannot be adopted by a new pool.

Reservations become active before worker creation or CUDA initialization.
Launching, running, and draining jobs retain their declared peak. Admission
adds active peaks, incoming peak, observed unattributed usage, and headroom.
Observed total usage plus headroom must also fit. Overruns, unrecognized
compute PIDs, ambiguous holders, UUID changes, and inventory errors block
new work. Process titles never establish ownership. Unsafe requests queue
in FIFO order or reject according to the explicit policy.

## Descendant and own-only stop safety

The audit required corrections before certification.

The author first found a leader-exit observation race. The final tick path
refreshes process and compute observations after poll confirms leader exit,
before considering completion or freeing the active memory reservation.

Independent scrutiny then found that the first kernel-child stabilization
check could accept lists read before the final identity merge. A child could
fork and exit during that merge. The final ProcReader reads every controller
task's kernel child list, merges fresh identities, rechecks the kernel lists
after the merge, and requires consecutive before/after equality. Instability
after eight attempts fails closed. The independent fixture forces that exact
child-to-grandchild transition and verifies continued drain/reservation.

A final independent fake identity check exposed reuse of an old leader's
numeric PID by an unrelated same-UID session. It failed on4068400e and is
preserved in pid_reuse.log/json. The final revision disables session ownership
when a current leader UID/start time differs from the saved leader. An absent
leader requires a live verified previously recorded member of the original
session. Parent ancestry also checks current UID/start correspondence.
The corrected independent PID-reuse test passes.

Unassigned controller descendants conservatively block drain/admission,
even after closing their lease FD. Ambiguous holders are not signalled merely
because they are ambiguous. SIGTERM uses only proven current UID/start-time
identities and pidfds. Actual Linux pidfd syscalls were tested against a
created CPU child. There is no unchecked PID/killpg fallback or SIGKILL
escalation.

## Operating-system threat model

This review covers accidental escaped own workers, process reuse, and foreign
CUDA conflicts. It does not require protection against adversarial same-UID
code transferring FDs to unrelated processes.

Known and launched descendants, kernel child lists, and their descriptors
must be inspectable. Inaccessibility fails closed and retains reservations.
Inaccessible unrelated CPU processes are reported without adoption or
signalling. Accessible unknown same-inode holders conservatively block work
even if their FD was independently opened. All foreign/unrecognized CUDA PIDs
block admission. Scientific workers must retain their inherited FD throughout
their CUDA lifetime.

Driver observations are snapshots. A foreign process can start after an
observation. The controller then blocks further work and never kills that
process. Controller loss preserves inherited kernel leases, but loss of the
stdout pipe can cause a worker to fail. Neither the stale registry nor a
replacement controller establishes successful completion.

## Storage and scientific fidelity

The shared persistent ledger uses the scheduler lock and one exact complete
hashed root snapshot, reserve, and ceiling binding. Its budget includes all
controllers and accepted jobs across pools, plus declared external writers.
The independent two-pool race admitted only one of two reservations that
could not both fit. Completed, failed, and cancelled outputs keep their
reservations. Output collisions, incompatible snapshots, free-space limits,
and storage ceiling violations reject admission. No silent rebase occurs.

Root's latest plan uses the existing complete snapshot,130 GB external reserve
for live work and legacy ARGMix, and134.4 GB of pooled job reservations.
Using root's reported167.337 GB baseline gives431.737 GB before the controller's
64 MiB reserve, approximately431.804 GB afterward. This is below450 GB.
These are root-provided planning figures, not an independent storage scan.
Root must pass the exact snapshot path/full hash and130000000000 external
reserve bytes, and ensure reservations cover all output growth.

The controller does not recursively measure arbitrary worker writes.
Scientific workers must enforce their own budgets. Pinned ARGMix and native
ARGformer still perform their existing startup root storage scans. They were
not modified for this scheduler.

The final allowlist includes biological candidate, GST, mechanism controls,
official ARGMix, native ARGformer training, and native ARGformer qualification.
Candidate/GST/mechanism use their unchanged lease/title/configure_fit path.
ARGMix and ARGformer preserve the descriptor through native exec and retain
their source guards. The unchanged NVTOP helper requires exactly the caller's
own PID/title match while allowing other same-device compute workers.

Scientific module/script/helper pins match. The pool passes literal worker
arguments without modifying seeds, models, configurations, preparation,
targets, losses, selection, support, or scientific inputs. Model/seed/output
fields are requested provenance. Root must supply exact reviewed arguments
matching that provenance and the output reservation. This remains source
compatibility evidence, not an actual model or CUDA qualification.

## Preserved checks and scope

author_replay_ae1e0494.log/json record25 passing author tests.
independent_ae1e0494.log/json record8 passing independent checks.
Earlier logs remain preserved. The first independent author replay exceeded
the Linux Unix-socket path limit because the reviewer used an overly deep
contained test root. It was shortened within this owned directory.
Two initial transition fixtures needed their string substitutions updated for
the author's expanded fake-worker branch. These were test-harness corrections.
No pool source was changed by the reviewer.

All review writes are confined to research/reviews/gpu_pool_v1_independent,
including fake roots and helper processes. No production GPU inventory,
lease, context, NVTOP call, model/native fit, scientific array, truth/outcome,
installation, simulation, Git operation, or existing-user-process signal
was performed. Real kernel process metadata and pidfd operations were used
only for the contained CPU lifecycle tests. No blocking issue remains for
this exact source revision under the documented boundary.
