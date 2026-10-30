# Queue sustainability repair, 30 October 2026

GPU 3 finished TreeLoom and native ARGMix predictions for IID000 and then had no ready GPU work. The next genealogy was still being reconstructed on CPUs. A serial scratch-release dependency unnecessarily prevented preparation of a second dataset despite available capacity. This was a scheduling bottleneck, not evidence of a failed GPU worker.

## Deployed

- A separate producer now owns exactly 48 previously untouched preparation stages for six remaining European TEST datasets. IID002 generation and Relate-input preparation have completed; its native reconstruction is running alongside the original IID001 reconstruction. A rolling window permits two unfinished parents and reserves enough storage for both complete chains before admitting work. Six permanent ownership markers prevent the old scheduler from launching duplicate generators.
- The native comparator follows the new owner for exactly twelve features/support dependencies. Its completed fit and IID000 prediction are reused through genuine completion receipts. No prior scientific worker was stopped or restarted.
- New producer and native controllers share a hardened scheduler. Unavailable filesystem, GPU or process observations pause relevant admissions. Genuine resource violations remain enforced. An actual child is registered before fallible status reads. Exit receipts are immutable and successful writes remain successful if their trailing verification read temporarily fails.
- Reporting follows the effective owner, preserves historical failed or claim-blocked rows separately, and reacts to changed waiting reasons and monitoring errors. Publication uses complete-file replacement. Finishing and final review follow the new reporting configuration.
- A reviewed ARGformer handoff watcher is running. It will replace only an exactly identified, naturally childless controller, after validating genuine waits and unchanged inputs. Existing TEST007/008 workers continue. If TEST009 has already started, the watcher exits without replacing anything. The replacement removes only TEST009 preparation's storage-cleanup dependency. It will not put a long nonpreemptible ARGformer fit on GPU 3 ahead of unfinished TreeLoom/native work.

## Validation and limits

Twelve common-engine tests, fifteen producer tests, eleven consumer/reporting tests and twenty-six ARGformer handoff tests passed. These include injected read/write failures, uncertain completion publication, duplicate prevention, changed ownership, wrong waits, storage completion floors, namespace restrictions, missed handoff boundaries and safe no-op exits. Synthetic queue tests launch no scientific jobs or CUDA work. Independent reviews accepted the final sources, followed by live activation checks.

Strict accounting reserves about 1.032 TB of the 1.2 TB project ceiling, with about 168 GB remaining. This is committed capacity, not a fresh full-directory disk-usage measurement. The original European family remains capped at 128 GiB. Full completion reservations and a 1 GiB guard govern the two-parent window. Author checkpoint leaf aliases are counted conservatively without traversing symbolic links; unrelated links and inaccessible directory walks are rejected.

GPU 3 will still be idle when every available GPU stage depends on unfinished CPU genealogy reconstruction. The repair overlaps preparation to reduce those gaps. It does not promise constant GPU utilization or run unnecessary experiments to inflate utilization. The original IID001 and ARGformer controllers remain unchanged while they own healthy workers; the new observation handling applies to the activated successor controllers.

No model architecture, simulation parameters, random seeds, observed data, splits, training budgets, scoring rules or registered comparator condition changed. The existing ARGMix equation-corrected condition remains disclosed. The European paired TEST score remains pending its registered prediction/scoring barrier. This operational repair produces no new accuracy claim.

## Work status

| Item | State |
| --- | --- |
| Remove avoidable serial European preparation | Deployed; IID002 progressing |
| Preserve current scientific workers and completed results | Verified |
| Harden new controllers against transient observations | Deployed and tested |
| Authenticate producer/consumer ownership | Deployed and tested |
| Publish actual blockers and separate historical failures | Deployed |
| ARGformer controller transition | Watcher active; waiting for a safe childless boundary |
| Complete remaining benchmark and paired scores | Still running/pending |

Activation receipts are under `critical_path/producer_v1`, `sustainability/activations`, and `backfill`. The implementation and test receipts are in those same directories and `common`.

## Operational instruction deviation

One helper mistakenly staged five small plaintext source files in `/tmp/argformer_backfill_20261030` on the Mac before transferring them to anogena. This violated the server-only file-write instruction. No benchmark computation or Git ran on the Mac. The exact paths are recorded in `backfill/preparation_audit_v1.json`. All subsequent edits and tests were performed on anogena; no files outside the project were deleted.
