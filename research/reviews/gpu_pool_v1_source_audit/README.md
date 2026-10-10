# Source-only pool audit

The owned controller, CLI, documentation and handmade tests passed AST parsing
and 25 contained CPU/fake-inventory tests. No real GPU query, production lease,
scientific launch, scientific input read, installation or Git operation was
performed by this authoring task.

The two critical compatibility discoveries were handled openly. The server
restricts FD inspection of unrelated same-UID processes, so launched/recorded
descendants remain strict while unrelated inaccessible CPU PIDs are reported.
Linux subreaping is verified to retain orphaned worker descendants. The fixed
project Python lacks optional pidfd APIs. A standard-library ctypes wrapper uses
real Linux pidfd syscalls on checked architectures, preserving PID identity
binding and avoiding kill/killpg fallback.

Tests prove atomic two-device admission against canonical locks, interoperability
with a contained old-style exclusive lease, the identical shared open-file
description through an offset visible to two separate workers, unchanged guard
and immutable metadata, memory reservation before CUDA allocation, rejection or
queueing of foreign PIDs/overruns, descendant and escaped-worker drain, controller
loss lease survival, bounded logs, persistent storage and responsive socket
supervision.

The exact final hashes, test evidence and limits are in receipt.json. Production
driver/NVTOP qualification and root activation have not happened in this task.
Pinned resources, launcher, guards, scientific workers and helper hashes match
their supplied baselines. AGENTS was read and remains untouched by pool authoring.
The root scheduling agent supplied a newer AGENTS authorization allowing pooled
workers before this work; its currently observed hash is recorded separately.

Independent review should assess Controller._process_scan, _memory, _launch_queue,
_signal_own, tick and run, acquire_device and StorageLedger. The production CLI
does not expose test injection. Consider the explicit storage contract, retained
worker startup scans, same-UID threat boundary and controller-loss limitations
before activation.

GST was added after root explicitly requested pooled admission. Source inspection
verified standard-library module imports, lease/title before scientific imports
and unchanged candidate configure_fit/check_fit_lease reuse. Both GST module
and trainer hashes remain pinned, and no GST config/dependency was edited.

Mechanism controls were added after root explicitly requested pooled admission.
Both module and trainer validate lease/title before scientific imports and reuse
the unchanged candidate configure_fit/check_fit_lease. Pinned scientific files
and configurations remain untouched.

The justified replay exposed a mixed /proc snapshot race in exit finalization.
The repaired controller refreshes process/compute observations after confirmed
leader exit and uses stabilized kernel task child lists to capture newly
subreaped descendants, including a descendant that forks/exits during a scan.
A deterministic stale-snapshot regression now preserves the reservation until
its inherited-FD child drains. The final suite of 25 fake-only tests passes.

Independent review identified the final identity-merge ordering boundary.
The barrier now rechecks kernel child lists after reading identities; a new
subreaped grandchild forces another pass. A deterministic worker creates a
child-to-grandchild transition at that exact merge boundary and confirms its
64 GiB reservation stays active until the grandchild drains. Both finalization
regressions pass in the final suite of 25 fake-only tests.

Independent review also found an accidental numeric leader-PID reuse hazard.
Session ownership now requires the original UID/start identity or an already
verified live original-session member, and is disabled on a mismatched current
leader. Parent ancestry requires the current parent UID. A handmade reused-PID
regression refuses adoption and retains foreign compute classification. All 25
fake-only tests pass; no real unrelated process was signalled.
