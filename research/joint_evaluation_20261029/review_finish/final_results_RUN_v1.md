# Running the independent final audit

Run on anogena from /workspace-SR003.nfs2/ARG-GNN. A single root-owned CPU controller may invoke this after research/joint_evaluation_20261029/root/finish_records_v1/ready_for_final_review.json appears. That marker is a trigger only: the audit independently validates the underlying successful waits and products.

```sh
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
.runtime/env/bin/python3.11 -B -s -u -m \
research.joint_evaluation_20261029.review_finish.final_results_audit_v1 \
--benchmark all
```

The successful immutable result is research/joint_evaluation_20261029/review_finish/final_results_all_completed_v1.json. The program never overwrites an existing result. For independent completion of either comparison, use --benchmark american or --benchmark argmix; their default filenames replace all with the selected benchmark name. --output accepts a new final_results_*.json file directly in review_finish only.

Use --check-ready for a read-only wait/product check. It opens no prediction arrays or private truth. Running the complete audit while pending also returns before opening truth or writing a result.

Exit codes:

- 0: requested numerical audits passed and the final result was written; with --check-ready, the scoring waits are ready without running a numerical audit.
- 3: a required completed scoring wait is absent. No truth was opened and no result was written.
- 1: invalid evidence, checksum mismatch, live worker session, numerical mismatch or another runtime failure. Preserve the log and inspect the failure; do not publish a PASS or silently replay a model stage.
- 2: invalid command-line usage.

American requires the root-owned paired-comparison wait. ARGMix requires joint and native TEST scoring waits and the separate joint validation-score wait. All child prediction and original scoring waits are checked again during the audit. The audit uses NumPy and tskit on CPU, reads one truth parent at a time, checks a32GiB process RSS limit, and caps the new report at4MiB. It imports no production training or scoring function.

Tests were run with:

```sh
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
.runtime/env/bin/python3.11 -B -s -u -m \
research.joint_evaluation_20261029.review_finish.final_results_tests_v1
```

The final test receipt is final_results_tests_result_v2.json. The first receipt is retained as historical draft evidence. Tests use synthetic ancestry truth only, plus original public American prediction arrays and original public TRAIN support. They do not replace the full numerical audit after production completes.

The scientific contract statement is final_results_contract_review_v1.md. Keep it separate from numerical PASS evidence. No publication or Git operation is performed by these files.
