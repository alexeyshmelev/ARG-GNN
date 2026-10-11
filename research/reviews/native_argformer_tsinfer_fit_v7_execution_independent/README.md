# Native V7 root queue V1 metadata review

Verdict: REJECT. The existing independent native source/config PASS remains valid. This is a separate review of the root queue/plan packet only.

Three fake-root counterexamples demonstrate missing exact-plan bindings: a changed empty natural-release barrier is accepted; a different activation-selected budget receipt is accepted; and an unreviewed stage proposal is activated. The native admission and original launcher remain intact and may reject invalid values. No bypass of those scientific or free-device guards is claimed.

Minimal V2 changes:
- Require activation.budget_receipt to equal plan.budget_receipt.
- Resolve/hash the exact plan.release_barrier before waiting or writing queue receipts.
- Resolve each unique (parent,stage) through the exact12 plan records; hash the proposal and registration, verify their mutual reference and scope/stage before activation.
- Refresh source_delta.json and bind a fresh current GPU2 natural-release barrier. Root reported different live GPU2 workers after V1 preparation.

The finite harness passed25 checks of normal scheduling, own-parent probe/fit receipt links, CPU retrieval overlap, all four probes first, stopping at each possible first failed probe, and process-identity matching/PID-reuse/UID-change behavior. It changes ROOT only in an in-memory AST prefix and uses an owned fake filesystem and external-action mocks. It never runs the real queue.

The saved packet passed114 metadata assertions and33 exact reference hashes:12 proposals/registrations, source/config and existing independent PASS, complete captured runtime inventory, GPU2 UUID and CPU role, Python/executable/helper bindings and the genuinely applied four20GB reservation receipt. Saved inventory metadata was read; no native prefix files were scanned. Original500GB measured launcher, free exclusive lease, process title and two-device cap remain. Launcher PYTHONPATH=src and the unchanged core's empty scientific child path remain.

Initial harness errors (aliased fixture budget ref and omitted retrieval-list initialization) are preserved. An uncounted auxiliary command fixture stopped on a harness argument-index assertion; its small own mock files are retained and it produced no claim or actual Popen. No runtime/data/model/GPU/budget/scheduling/Git operation, live process observation, queue activation or source edit occurred.

Root must preserve this candidate and review a fresh immutable V2 before queue activation.
