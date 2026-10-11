# Independent full-reservation launcher source check

**PASS**, narrow exact-source review, for scripts/launch_reserved_gpu_full_storage_v1.py SHA256 `d7e5b2bcb464b9d3863b4f644e69aaa1c73c18aaa2550e44ecd021e2199f6dff`.

Original v6 is unchanged at SHA256 `a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3`; V9 core is unchanged at `483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e`.

The exact diff changes the description, adds the embedded locked full-charge helper, replaces the two storage checks, labels the lease's accounting honestly and gives exit evidence a distinct filename/kind plus accounting metadata. The main AST matches original v6 after normalizing only the storage-accounting label. GPU selection, two-device cap, eligible/free checks, locking, UUID mapping, inherited lease FD, consumer environment, process launch, signal forwarding, wait and lease cleanup remain unchanged. Existing consumer process-title/scientific/runtime behavior is reached through the same module command/environment.

The new storage helper verifies its exact current source reference in the reservation's consumer_bindings, while preserving the original v6/core dependency bindings. It requires the full-reservation accounting mode, equal record/reservation allowances and the exact own output path. The original locked V9 checker still verifies immutable authority, journal and active reservation with boundary=False. That disables global closed-scope process/writer inspection; the helper then restores full charges as v4_projected_bytes + v9_reserved_bytes, without subtracting closed allowance credits, and rejects a charge at or above the exact human 1,000,000,000,000-byte limit.

Only the current reservation's canonical output/auxiliary scopes are stat-measured. Measured bytes plus planned exit-evidence growth must fit the original allowance. The helper acquires no mutex; both original main mutex sites remain, and both storage-call replacements execute under them. This is contractual reservation accounting and own-scope checking, not a filesystem quota or a new claim about unrelated external writers.

This review executed no launcher, helper, admission, worker, GPU or model tests. Source files were read and parsed only. No runtime reservation/data/ledger payload was opened. Current budget admission, GPU availability and scientific execution qualification remain root's execution responsibilities. The earlier bootstrap review is preserved unchanged.
