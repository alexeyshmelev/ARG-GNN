# Completed candidates: preliminary retained-storage inventory

Readonly proposal only; no settlement authorization or apply. Writes were confined to this audit directory. Completed outputs/logs, live ledger, controllers and decision records were unchanged. No arrays/labels or output receipts were decoded; no GPU query/context was created.

| Seed | Files + directories | Retained stat charge | Proposed charge | Growth margin | Reduction |
|---|---:|---:|---:|---:|---:|
| 202 | 201 files + 2 directories | 10,213,036,034 B | 15,000,000,000 B | 4,786,963,966 B (46.9%) | 14,281,048,576 B |
| 303 | 212 files + 2 directories | 10,692,829,186 B | 15,000,000,000 B | 4,307,170,814 B (40.3%) | 14,281,048,576 B |

Every retained file/directory and pool log is charged using max(st_size, st_blocks * 512). Both complete stat/membership passes matched. Terminal jobs remained succeeded with exit code 0; all four saved PID/start identities were absent at both checks. This does not prove absence of arbitrary unknown writers.

Fresh live ledger SHA256: ddfdd905b645f48fbd7fcb7cafb3fc2c86bee801cc923abe37ba81d2b1d09e26. Recomputed current projection: 448,490,318,848 B. Proposed combined reduction: 28,562,097,152 B; proposed projection: 419,928,221,696 B; headroom: 30,071,778,304 B under the 450,000,000,000 B limit.

The fresh ledger includes both 1,000,000,000 B census reservations. Preserve all unrelated jobs, all four 67,108,864 B controllers, historical baseline 167,336,640,512 B, external reserve 130,000,000,000 B, limit, snapshot and output bindings.

Proposed charges have a 15 decimal GB floor and at least 25%/2 decimal GB measured-growth margin, rounded up to decimal GB. All epoch probability NPZs remain included. Future fits need fresh output directories and separate reservations.

## Minimal delta to the reviewed GST settlement

1. Bind both exact candidate keys, roots, retained logs, registry terminal jobs, seeds and all saved PID/start identities; replace GST hardcoded job/status/ledger bindings.
2. Verify candidate receipt metadata only: kind biological_candidate_fit_v1; status completed_three_conditions_one_seed; matching seed/source/config/observed/prepared pins; exact expected three model keys with completed and optimizer_fit_completed=true. Do not decode arrays or labels.
3. Freeze these stat artifacts as prior evidence and create opaque SHA256 file bindings for every output file and retained log with before/after stat checks. Stat-only evidence cannot substitute for the reviewed full-hash inventory.
4. Require root contractual freeze of completed outputs/logs and trustworthy terminal registries/cooperating ledger writers. Under canonical .runtime/locks/scheduler.lock, create two full inventories, recheck terminal jobs/PID-start identities before and after, bind the fresh ledger, then use the reviewed atomic mutation.
5. Independently review the candidate adaptation before apply. Change only reserved_bytes for the two existing bound keys and assert exact preservation of every other ledger field/reservation. Retain completed keys permanently.
6. Retain all files, including every epoch probability NPZ, checkpoints, metrics and logs. Future fits use fresh output roots and reservations. Preserve mechanisms, compact workers/controllers, historical baseline and 130000000000-byte external reserve.
7. Do not run the old hardcoded GST script for candidates or claim OS immutability/arbitrary unknown-writer absence. Preserve the qualified PASS contract explicitly.

## Evidence and limitations

Pins are recorded in inventory_receipt.json. Candidate source establishes the receipt schema at lines 733-750 and terminal completion at 867-880. Actual output receipt metadata verification remains outstanding because this preliminary task reads only terminal registries and filesystem stats. Opaque file hashing and two full inventories under the canonical mutex remain required before applying an independently reviewed adaptation. Contractual freeze and cooperating-writer assumptions remain explicit; the procedure does not enforce OS-level immutability.
