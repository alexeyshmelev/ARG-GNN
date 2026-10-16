# Prospective1611 TRAIN000: active-writer boundary failure

The saved failure is a parent-side storage-snapshot rejection, followed by an owned SIGTERM and genuine drain. The wrapper applies a quiescent-snapshot check while the generation child is allowed to write. Normal output growth can violate that check without exceeding a byte cap or changing the science. The saved receipt does not identify which path changed, so the particular file and write actor remain unknown.

## Confirmed mechanism

The actual wait records child 3449313/start 482094439, elapsed 120.537 s, ValueError “scope stat/membership changed during boundary”, direct worker return -15, peak combined RSS 259616768 bytes, and all owned processes/groups drained. The parent supplied actual outer tool session 42406/exit 1/chunk 848541. No worker traceback or progress log was captured. Empty output is not evidence that numerical work never began: the generator prints its completion only at the end.

The exact call chain is:

1. root_generate_v1.py:226–257 supervises the still-active direct child. It checks local scope caps every approximately 0.25 s, then calls guard.budget_check_v9(storage,boundary=True) every 60 s at line 244.
2. The unchanged H020v3 guard takes the accounting mutex and authenticates the open reservation, exact source bindings and journal prefix. At line 511 it asks v4.measure_scopes to measure the owned output root and worker_aux.
3. project_budget_v4.py:250–276 recursively records each path, then restats all recorded paths. Equality includes device, inode, mode, link count, file size, allocated blocks, mtime_ns and ctime_ns. The exact saved message is emitted at line 275 by the final restat pass. It is distinct from the immediate directory-listing error at line 269.

Thus at least one recorded stat changed between its first sample and final recheck. Ordinary append growth changes size/blocks/timestamps; new expected files can change directory size/timestamps. These changes are enough to fail, even when namespace identity and the full 16 GiB reservation remain valid. The accounting mutex does not suspend the generator or serialize its writes: generation.py writes directly through exclusive file opens, streaming text writes and native truth.dump.

The source intentionally creates observed/private/receipt directories (line 365), writes metadata (383–387), dumps simulation.trees (388), then streams observed VCF/HAPS/TSV files (export_observed, especially 316–344). OutputBudget.write enforces the observed byte limit before each text write (296–301); the wrapper separately sets RLIMIT_FSIZE and monitors resource/scope caps. No changed path, before/after values or traceback was saved, so the failure cannot prove which of these writes overlapped the scan, or rule out an unsafe mutation. Its timing is consistent with a periodic guard boundary; it is not an observed RSS or wall-limit breach.

After any monitor exception, the wrapper deliberately sends SIGTERM through owned pidfds (257–283), takes the real OS wait and proves drain. The -15 return therefore records the termination response to the operational failure; it is not a scientific exception result. Success-only completion is rejected at 421–422. Preserve this failed scope and charge: there is no successful completion, retry, resume or credit claim.

## Is the stability requirement appropriate?

Yes for a true quiescent prelaunch/post-drain boundary or completed-scope close. It detects changed namespace identities and gives one coherent allocated-byte measurement. It is inappropriate as an unconditional “nothing may change” requirement while an admitted streaming writer runs independently. The current call site conflates expected output growth with corruption. This is a wrapper/checkpoint-placement mismatch; it does not justify weakening the sealed global guard.

## Minimal scientifically unchanged repair possibilities

The smallest future call-site change is to use the unchanged guard with boundary=False only for the periodic live authority/reservation/prefix check, retaining boundary=True before releasing the child and after genuine wait/drain. Retain the existing 0.25 s resource/output checks, OutputBudget and RLIMIT_FSIZE. Before approving that option, review its live measurement contract explicitly: use conservative max(st_size,st_blocks*512) and directory allowance, retain containment/type/inode/link/expected-file checks, preserve the full charged 16 GiB cap and margins, and document that sampling is not an OS quota. This changes the periodic operational safeguard, so it needs a fresh reviewed wrapper and root authority; it is not a silent bypass or an authorized retry.

If the exact full boundary must remain every 60 s, a narrow wrapper alternative is to quiesce the one owned child through its held pidfd, establish the real stopped state and unchanged owned roster, run the original boundary while writes are quiescent, and resume that same identity. Keep deadlines, cancellation handling and stop/drain guarantees. Pending filesystem metadata may still need bounded settling/rechecks while stopped. This preserves the guard body and scientific seeds/equations but adds lifecycle handling that must be reviewed before use.

Do not broadly catch-and-ignore this ValueError or substitute an old measurement: the same message can also represent inode/type/link/namespace replacement. Bounded blind retries alone cannot ensure progress during sustained output growth. A future wrapper should record the changed path and exact differing fields in bounded failure metadata; the existing receipt cannot reconstruct them.

All scientific interval, cohort, map, demography, seeds and error realization remain unchanged under either operational repair. Neither option is implemented or authorized here. This diagnosis read only existing Python/text/JSON metadata and the saved empty-log size; it loaded/hashed no scientific payload, imported/executed no project or scientific module, performed no process/GPU probe or accounting call, edited no sealed/live source, and launched/retried nothing.
