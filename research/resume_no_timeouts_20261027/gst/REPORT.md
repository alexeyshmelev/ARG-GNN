# GST resume from durable update 79

The existing source already supports exact complete-parent recovery. The actual checkpoint passes validation at **79 of 90 AdamW updates**, and the unchanged fit loop passes an interrupted-versus-uninterrupted synthetic smoke across that same boundary. The staged replacement resumes the remaining 11 updates; it preserves the original model, data, parent order, optimizer settings, four workers, 30-epoch budget, and fixed final-epoch selection.

No failed output was renamed, no request was published to the live queue, and no production fit was launched. All preparation and smoke outputs are under this directory; no Git changes or running-job changes were made.

## Actual checkpoint validation

The checkpoint is `experiments/results/paper_gst_lai_8_20261024/paper_gst_lai_8__all_train__fit/native/recovery_latest.pt`, 639,404 bytes, SHA256 `02a19bd79e68aabd01d2dd3bbbbd9c6e333470961ab0512b199466995684de19`.

`checkpoint_validation.json` records strict model-state loading, complete AdamW restoration, finite model/moment tensors, all 80 optimizer counters equal to 79, and 79 valid history rows. Saved source pins match the current original sources exactly. Original plan, class order, TRAIN input bindings, deterministic shuffled parent prefix, and observed-input proof match the saved observation barrier.

Epoch 27 has completed only `ancestralpaths_european_train_000`. Resume starts with TRAIN001 at update 80, then TRAIN002 at update 81, followed by all three original parents for epochs 28–30. The full remaining schedule is in the validation artifact. Any uncommitted partial-parent work after update 79 is recomputed without crediting or repeating a committed update.

The checkpoint stores Python, NumPy MT19937, Torch CPU, and one CUDA generator state. CPU inspection verified their structure without initializing CUDA. The production source builds the model on its admitted single GPU, restores model and optimizer state, creates the new loader iterator, then restores RNG to undo the extra worker-seed draw before resumed dropout. AdamW uses the unchanged constant learning rate 0.001; there is no separate learning-rate scheduler to reconstruct.

The checkpoint is archived unchanged, including its historical observation-barrier reference. On resume the original source reauthenticates original observations, compares their proof with the checkpoint, and writes a fresh barrier. It does not dereference that historical output barrier while restoring. Full observations were not reloaded during this preparation.

## Meaningful interrupted-run smoke

`resume_smoke.py` invokes the actual unchanged GST fit function, substituting only small synthetic observations, TRAIN labels, and their plan. It uses the original model, dropout, complete-tree encoding, four persistent spawned workers, 30 epochs, 90 updates, checkpoint save/load, and recovery validator.

An uninterrupted run was compared with a run deliberately interrupted immediately after durable update 79 and resumed for the remaining 11 updates. In **40.97 seconds**, final model tensors, complete AdamW state, Python/NumPy/Torch CPU RNG, every scientific history row, full visit counts, and epoch/parent progression matched exactly. Elapsed clock fields were excluded from equality. All 80 learned tensors reached counter 90. This CPU synthetic result does not claim full-data CUDA bitwise equivalence.

Receipt: `smoke_v1/receipt.json`. The log and all synthetic checkpoints remain available.

## Inert queue repair packet

`replacement_tasks.json` and `repair_request_draft.json` contain the fit plus its two currently blocked downstream tasks. Only two kinds of fields change: each task's `max_seconds`, and the fit's `resume_checkpoint_ref` pointing to the proposed archived recovery file. Original scientific/runtime source-reference lists and all memory, storage, data, output, dependency, and GPU restrictions remain exact.

The existing live controller requires a positive integer wall limit and cannot be replaced without disturbing healthy workers. Therefore **`max_seconds=9223372036854775807`** is the explicitly reviewed compatibility sentinel for effectively unlimited total wall time. It is not a new 24/48-hour cap. Future controller code may support a null value directly. The current source's `DataLoader(timeout=1800)` is retained as a per-batch stall detector, distinct from a total fit wall limit. It should be revisited only if productive batches are measured to exceed it.

`protocol_preflight.json` reconstructs the current effective plan through applied repair receipts, verifies exact source pins and the limited field changes, and passes the current scheduler validator with the sentinel. At preparation, plan revision was 4 and all three affected attempts were 3. The failed owned process session was absent; the two downstream output directories did not exist. The live controller's normal repair protocol will advance all three to attempt 4.

`archive_plan.json` pins all **31 files / 680,491 logical bytes** in the failed fit output, including original inode, timestamps, size, link count, and SHA256. The intended whole-directory rename is:

- From: `experiments/results/paper_gst_lai_8_20261024/paper_gst_lai_8__all_train__fit`
- To: `research/resume_no_timeouts_20261027/gst/attempt3_archive`

ROOT must freshly verify the failed state/session and every pinned file, perform the rename once, verify destination hashes, and write a final archive receipt. The draft currently references the **plan**, not completed archive evidence; ROOT must replace that reference with the final receipt, refresh revision/attempt bindings, and publish the exact named `.request.json` followed by `.ref.json` in the live `repair_requests` directory. Original attempt-3 job and log remain in place. Preserve the healthy TreeLoom and ArgFormer processes throughout.
