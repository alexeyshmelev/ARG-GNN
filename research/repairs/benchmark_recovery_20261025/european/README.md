# European AP TreeLoom prediction repair

The AP final fit completed all 30 epochs and 90 optimizer updates. The epoch-30 checkpoint is usable: original completion and checkpoint validators passed, all 49 model tensors are finite and all 49 Adam counters equal 90. The fit remains succeeded and is reused without retraining.

The failed held-out prediction called the raw full-length genomic GRU and exited with cuDNN CUDNN_STATUS_NOT_SUPPORTED. The generic contiguity hint does not establish non-contiguity as the cause. The completed fitter had already installed the existing bounded GRU adapter; its prediction wrapper omitted that installation.

The isolated ap_prediction_chunked_gru.py wrapper installs that same existing adapter, with 1024 holds per connected bidirectional chunk, on the newly loaded prediction model. It preserves the weights, every original observation, sampled identities, complete trees and history, six output classes, product schemas and resource limits. The task retains its 512 GiB host cap, 128 MiB output cap, 56 GiB minimum free GPU requirement and GPU authority 3/4/5. Original sources are unchanged.

## Evidence

- checkpoint_validation.json authenticates the 443,754-byte epoch-30 checkpoint and completed fit.
- cpu_smoke.json records actual saved weights on a non-contiguous [2051,2,35] input, nonzero incoming state and three chunks per direction. Output and final-state maximum errors were 1.79e-7; sampled-logit error was 5.96e-8. Parameter identities, state keys and RNG were preserved. No CUDA, optimizer creation or truth access occurred.
- ../ap_cuda_smoke.json records ROOT's small GPU 5 smoke: output error 1.69e-7, final-state error 1.19e-7 and 56,623,104 peak reserved bytes. No training or checkpoint modification occurred. This is recurrent parity evidence, not a full-parent prediction or ancestry-accuracy result.
- preflight_read_only.json records a successful read-only preflight at campaign revision 6: 171 replacement source pins, unchanged prediction specification and task resources, exact effective-plan replay, valid candidate scheduler plan and current failed-session absence.

## Archive and request

archive_plan.json describes exactly five failed-scope files totalling 77,803 bytes. There are no prediction products. The original failure log remains in place and is charged separately. Preparation has not moved the failed scope.

inert_request_draft.json and replacement_task.json are research-only drafts. The draft archive_manifest_ref is null until explicit apply creates the exact archive receipt; the draft is not a live controller submission.

apply_repair.py defaults to read-only:

```bash
cd /workspace-SR003.nfs2/ARG-GNN
PYTHONPATH=src:. .runtime/env/bin/python3.11 -B -u -m research.repairs.benchmark_recovery_20261025.european.apply_repair --expected-plan-revision 6
```

Explicit --apply revalidates pins and the failed session, archives the exact failed output scope, checks actual repairs._prepare against a research-only inert pointer, then publishes immutable request/body references for the existing controller. It resets only paper_ap_treeloom__all_test__predict and its blocked paper_gst_lai_8__all_methods_all_test__score descendant. It does not start a worker or signal a process. The common score continues to wait for its other prerequisites.

## Deferred scheduling

The next native Relate stage may grow from a 16 GiB to a 256 GiB allowance. To avoid cancelling the running full-parent profile, prediction submission is held until the full_gpu5_v2 child has actually exited.

deferred_submitter.py is default read-only and pins the apply helper, manifest, exact profile launch, profile supervisor and session inspector. Its explicit mode waits at most five hours for a matching actual_wait.json with owned_group_absent=true plus a fresh read-only session-absence check. Either profile success or failure permits submission after exit. It then invokes the frozen apply helper with the current integer campaign revision. The existing controller retains resource admission. There are no process signals or direct GPU launches.

ROOT may launch the reviewed waiter once:

```bash
cd /workspace-SR003.nfs2/ARG-GNN
PYTHONPATH=src:. nohup .runtime/env/bin/python3.11 -B -u -m research.repairs.benchmark_recovery_20261025.european.deferred_submitter --submit-after-profile --wait-seconds 18000 --poll-seconds 30 > research/repairs/benchmark_recovery_20261025/european/deferred_submitter.supervisor.log 2>&1 < /dev/null &
```

Its request, status, submission, apply stdout and result are retained in deferred_submission/. The supervisor stdout is deferred_submitter.supervisor.log. Existing deferred_submission/ prevents an accidental second launch. Root owns launch, final review and Git publication. This preparation did not archive, submit, launch, signal, commit or retrain anything.
