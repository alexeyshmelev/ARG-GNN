# V7 closed-scope accounting

V7 is a fresh append-only successor at .runtime/storage/project_budget_v7.
It preserves V4 accounting, original supplemental reservations, V6 source and
review bytes, and the original resource authorization. V6 must never have been
activated: every V7 root admission requires
v6_never_activated_root_attested: true, and the implementation rejects any
nonempty predecessor journal. V7 admissions require
all_new_consumers_use_v7_guard: true; the V4-versioned GPU launcher requires
consumer_uses_v7_guard: true in a V7 storage record. New reservations pin the
V7 consumer adapter and exact source bytes.

The only new process-observation exception is a root-pinned record containing
at most two opaque credential daemons, one per role: an external ssh-agent that
started before the pinned project start, and a root-created project
credential-only ssh-agent. Both require exact PID, kernel start ticks, comm
ssh-agent, current-user UID, a nondumpable attestation, and an explicit
attestation that scientific scope-write authority and relaunch have not been
delegated. Ownership is an attestation by the root agent, not a requirement that
the Linux UID be zero.

The project role additionally binds the canonical creation environment by
SHA-256 and the canonical UNIX socket by device, inode, UID, and socket type.
Environment bytes are hashed without decoding or output. The socket is never
connected to, and no credential contents are inspected. The external role
requires public kernel boot/start metadata to establish that it predates the
project. A bounded, complete, root-attested inventory of recorded scientific
PIDs must include every closure terminal PID; no listed daemon may be any
recorded scientific worker or launcher.

Only a PermissionError while reading an exact listed daemon's cwd enables the
opaque branch. The daemon identity and applicable environment/socket binding
are rechecked before inspection, around that branch, and before return.
Permission errors from other own-user processes, descriptor inspection, or
mapping inspection retain the inherited fail-closed behavior. Unlisted,
changed, other-user, duplicate, non-ssh-agent, and recorded scientific identities
cannot receive the exception. The branch records that descriptors/maps were not
inspected and sets local_proc_inspected false.

Closure still requires original reservation/completion bindings, terminal PID
absence, original GPU lease identity and exclusive ownership, scoped stable
measurement, exact independent review, scheduler mutex, future scope-write and
relaunch revocation, and repeated checks at guarded write boundaries. A closed
reservation changes its allowance to its measured conservative retained size
exactly once. New storage diagnostics use v7_reserved_bytes; actual supervisor
wait evidence uses launcher_exit_v4.json and
reserved_gpu_v4_supervisor_wait.

This is cooperative project accounting with two explicitly opaque credential
interfaces. It is not a global proof of absent writers: remote NFS writers and
external processes remain outside the observation claim. Root activation,
scientific execution, GPU scheduling, and Git operations remain separate from
this source-only author packet.
