# Project storage cleanup, 27 October 2026

Completed user-authorized cleanup only within `/workspace-SR003.nfs2/ARG-GNN` on anogena. No project files were written or deleted on the Mac. No directory or file outside the repository was deleted.

**Removed 2,962 obsolete files representing 380,588,158,976 allocated bytes (380.59 GB).** Logical file sizes total 379,076,164,571 bytes. Every selected file is absent. Each was a regular file with one link, and its recorded device, inode, size and modification time matched immediately before unlinking.

| Removed category | Allocated GB |
|---|---:|
| Historical generated data and tensor caches | 153.23 |
| Historical epoch predictions and obsolete pilot tensors | 134.57 |
| Downloaded installer and pip caches | 6.60 |
| Superseded C4 ARGMix training archive | 86.18 |

## What remains

Code, design history, all metrics and text records, manuscript files, configurations, installed environments and the current benchmark inputs/checkpoints remain. We also retained 50 selected prediction/readout archives and the small historical best-epoch checkpoints. Identity checks confirmed that 3,633 explicitly retained files in affected scopes stayed unchanged. Uncertain historical dependencies were left in place.

The deleted tensor and data files can no longer be read directly by old pilot scripts. Revisiting those retired conditions requires regeneration from the retained configurations and source. Historical records are preserved as records of the original runs, not rewritten to claim that deleted artifacts still exist.

## Validation and current work

Independent source review and 15 focused safety tests passed. Both deletion phases passed dry runs and completed with actual exit 0. No tracked file was deleted. No symlink was followed by the deletion tool. Twenty-five links in current workflow scopes were checked and none targeted selected files. Live descriptor/mapping checks were repeated during deletion. The only inaccessible process exceptions were seven identity-bound SSH agents. These are sampled live checks, not atomic exclusion of arbitrary concurrent writers.

The main queue had 49 succeeded, 200 pending, 3 running, 0 failed and 0 blocked at the post-cleanup snapshot. The same TreeLoom, ARGformer and GST worker PIDs remained present. No process was signalled or restarted.

The separate full 200Mb European validation worker had already failed with an aggregate-monitor FileNotFoundError before this cleanup. Its actual wait is 1 and its failure record predates cleanup by 8,064 seconds. Its files and both earlier short-validation records remain intact. This auxiliary failure is separate from the main queue counts and needs a later runner repair.

## Size accounting

The earlier full inventory measured 924.16GB. Subtracting these removals gives 543.57GB before subsequent benchmark writes and audit records. This is an estimate, not a new complete size measurement. No second expensive project-wide scan was run. Scheduler reservation journals were not rewritten and no extra benchmark was launched using hypothetical capacity.

## Records

- [Compact result](summary.json)
- [Dependency audit](fidelity_audit.md)
- [Execution and archive audit](execution_audit.md)
- [Selected historical predictions retained](fidelity_selected_evidence_v1.json)
- [Independent cleanup tool review](INDEPENDENT_CLEANUP_TOOL_REVIEW_v2.json)
- [Phase1 exact manifest](phase1_reviewed.json) and [completion](phase1_reviewed.completed.json)
- [Phase2 exact manifest](phase2_reviewed.json) and [completion](phase2_reviewed.completed.json)
- [Phase1 deletions](phase1_reviewed.deletions.jsonl) and [phase2 deletions](phase2_reviewed.deletions.jsonl)
