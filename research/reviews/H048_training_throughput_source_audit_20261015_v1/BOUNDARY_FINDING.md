# H048 pass-boundary finding

At 2026-10-15 16:51:19 and 16:51:52 UTC, the actual H048 PID 2912011 was sleeping with FD 63 open on `.runtime/locks/scheduler.lock`. Its I/O counters were unchanged across 33.10 seconds; user CPU increased one tick and system CPU did not change. FD 63's second `fdinfo` had no held-lock entry. This supports a pending scheduler-mutex acquisition as the current gap's location. The exact blocked frame and mutex owner were not observed by this worker.

The last recorded pass was 12: `TRAIN_pass_12.json` mtime 16:37:32.898713 UTC, recorded loop time 234.5464 seconds, 384 updates per arm. Pass 13, completion and failure receipts were absent. Registered checkpoints are passes 5, 10, 15, 20, 25 and 30, so the gap after pass 12 is not a scheduled checkpoint save.

The source path explains how a small receipt can wait outside the recorded compute time:

- `train_marker_grid_reset_chain_v1.py:234,258–260` starts the pass timer before the parent loop, captures seconds before `output.save("TRAIN_pass_%02d.json")`, then saves. Initial observation preparation and following checkpoint saves are outside that timer. A save of pass 13 may be waiting before file creation; absence of pass 13 does not prove no pass-13 computation occurred.
- `prospective_980gb_v1.py:436–450` serializes a save and calls `budget_check_v9` before creating its output pathname at line 453.
- `prospective_980gb_v1.py:417–418` enters `budget.v4.mutex(ROOT)`; `project_budget_v4.py:109–114` opens the scheduler mutex and performs a blocking exclusive flock.
- The installed transaction memo wraps the original mutex at `transaction_memo_adapter_v2.py:143`; it cannot cache or avoid waiting for acquisition. Its memo is local to one original mutex operation and is cleared on exit (lines 144–156).

The per-update finite-check host reads documented in the ongoing audit are a separate steady-loop optimization candidate. They do not explain this observed sleeping interval. GPU occupancy/utilization snapshots alone do not identify its cause. No lock owner was inspected by this worker, no signals were sent, and no payload, stack, syscall, GPU profile or checkpoint content was read.

Exact source/receipt hashes and a fresh permitted process/FD metadata snapshot are in `boundary_evidence_actual_v1.json`. Root owns lock-owner diagnosis and any intervention.
