# American benchmark continuation, 2026-10-26

The owner authorized continuing the benchmarks after the 2026-10-25 pause.
This package prepares two continuations; it does not launch jobs, signal any
process, reset queue rows, change the campaign plan, or overwrite interrupted
native output. All files and paths are within `/workspace-SR003.nfs2/ARG-GNN`.
ROOT remains responsible for admission on physical GPUs 3, 4, or 5, enforcing the
1 TB project storage cap, and preserving the stopped wrapper receipts.

## Exact restored boundaries

| Task | Durable saved state | Next work | Fixed endpoint |
|---|---|---|---|
| American TreeLoom fit | 238 complete-parent updates, epoch 12; all 49 AdamW parameter states | Update 239, parent `argformer_american_train_009`, epoch 12 | Update 600, epoch 30; 362 updates remain |
| Native ARGformer validation parent 001 MLM | 119500 batches, epoch 2, batch-in-epoch 29832; all 77 DecoupledAdamW parameter states | Update 119501, the next deterministic sampler batch | 224170 updates, 5 epochs of 44834 updates; 104670 updates remain |

TreeLoom source already supports strict model, optimizer moments, Python/NumPy/
PyTorch CPU/CUDA RNG and schedule restoration. The actual checkpoint was loaded
on CPU, its saved state and all optimizer counters validated, the optimizer
restored on CPU, and the existing observation barrier and parent schedule
validated. The original 20 TRAIN parents, full 30 Mb sequences, model, losses,
30 passes, and fixed final checkpoint remain unchanged. Prior epoch-5 and epoch-10
checkpoint event references remain in the saved history; the old epoch-5 file was
already rotated by the previous run and is an event hash, not a retained file.
The stopped update238 checkpoint and retained epoch10 file stay untouched.

ARGformer uses the unchanged author training entrypoint and existing paper-aligned
adapter. The only resolved configuration changes are `load_path`, explicit
`load_weights_only: false`, and the new `save_folder`. Five epochs, batch512,
8 persistent workers, seed17, literal 30 percent MLM masking, precision, optimizer,
learning-rate schedule and corpus identity are unchanged. Native Composer restores
the model, optimizer, scheduler, timestamp and main-process RNG. Its deterministic
PCG64DXSM sampler and default dataloader spin skip the 29832 already consumed
batches of epoch2. The wrapper explicitly requires that behavior and validates
the restored timestamp, all optimizer counters and scheduler before fitting.
The author `restart_override` is absent, so no LR, weight decay or clock reset is
introduced.

The eight persistent masking workers' RNG states were not serialized. Native
Composer restores the process RNG but cannot reconstruct those exact worker RNG
streams. Therefore sample order, cursor, scientific protocol and full budget are
preserved; an identical uninterrupted random mask sequence or bitwise trajectory
is not guaranteed. No attempt is made to invent missing worker states.

## Prepared files

- `treeloom.replacement_task.json`: complete replacement of `paper_american_treeloom_fit`.
- `argformer.replacement_task.json`: complete replacement of `paper_american_argformer_american_validation_001_mlm`.
- `treeloom_resume.py` and `argformer_resume.py`: readable callable adapters, with no launchers.
- `treeloom_evidence.json` and `argformer_evidence.json`: exact input checkpoint/configuration/source references and stopped-state provenance.
- `*.stopped.json`: preserved failed task rows and job references.
- `treeloom.qualification.json` and `argformer.qualification.json`: CPU-only validation receipts.
- `prepare_metadata.py`: the reviewed preparation source; it writes new files exclusively and intentionally refuses to overwrite its receipts.

ROOT may use the prepared task definitions with its explicit continuation support.
The original task output directories and resource limits remain unchanged. Existing
scheduler failed-branch repair machinery requires an absent old output directory,
so these tasks require ROOT's separate reviewed continuation support rather than
ordinary failed-branch repair submissions. Do not move either original native
output or reset the paused row without preserving its real exit receipt.

The new output leaves will be:

- `experiments/results/paper_american_20261024/treeloom_fit/native/resume_20261026_update238/`
- `experiments/results/paper_american_20261024/argformer_american_validation_001/mlm/native/resume_20261026_batch119500/`

After full-budget success, the adapters exclusively create the previously absent
original `native/training_completed.json` or `native/completed.json`. This retains
the existing downstream dependency paths. The original configs, progress logs,
checkpoints and checkpoint symlink remain unchanged. Composer checkpoint retention
rotates only its new `saved_checkpoints` list; loaded historical checkpoint event
references are separate and do not cause deletion of old files.

Retaining the old ARGformer checkpoint adds approximately 2.015 GB within the
existing 8 GiB task output reservation. Native keep1 rotation can transiently
retain two new checkpoints, so three total checkpoint payloads are approximately
6.045 GB. Checkpoint files remain subject to the original task reservation and
ROOT's global project accounting. TreeLoom retains its original 80 MiB output
reservation and 768 GiB RAM reservation; ARGformer retains 8 GiB output and 128 GiB
RAM reservations. No scratch dataset or cache is copied by this package.

## Validation performed

Preparation used visible direct SSH commands and plaintext Python source on
anogena. The submitted source was parsed with `ast.parse`; new files contain no
encoded commands, decoder/executor loader, subprocess launcher or shell execution.
The selected existing worker, native author Trainer call, Composer restore/spin/
checkpoint-retention implementation, TreeLoom restore path and IPC bootstrap were
reviewed as readable source. The IPC helper only allocates Linux abstract socket
addresses; it does not transport or execute an encoded program.

Both preparation modes completed successfully. The ARGformer checkpoint was
hashed once with explicit ROOT authorization; tensors were inspected using CPU
memory mapping. A wrong checkpoint cursor was rejected. TreeLoom exact saved
model/optimizer and schedule checks passed. No scientific dataset payload was
opened, GPU used, training update executed, job launched, or queue changed.
These are preparation and source-level restore checks, not a claimed completed
native GPU resume. Runtime guards will record `loaded_resume_state.json` after
ARGformer actually restores its Trainer, before any resumed optimization.

Commands used for the two bounded metadata preparations, from the project root:

```sh
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -s research/operations_resume_20261026/american/prepare_metadata.py --kind tree
CUDA_VISIBLE_DEVICES='' .runtime/conda/envs/argformer_native_v1/bin/python -B -s research/operations_resume_20261026/american/prepare_metadata.py --kind arg
```

These are recorded historical preparation commands. Their outputs now exist;
do not rerun them as launch commands. Use ROOT's scheduler with the prepared task
definitions and attempt-specific job/log records.

## Parent-specific downstream retirement

Use `argformer_retirement.replacement_task.json` for
`paper_american_argformer_american_validation_001_retire`. Its readable adapter
`argformer_retirement.py` preserves the original retirement implementation except
for the narrowly documented changes in `argformer_retirement.diff`:

- It accepts this parent's nested resumed MLM executed-spec directory, while all
  producer completion and actual-wait paths remain the original paths.
- It selects resumed MLM checkpoints from the new nested leaf for retirement.
  The interrupted `native/checkpoints/ep2-ba119500-rank0.pt` is explicitly retained.
- Its retained nonmodel bound is 2,149,128,390 bytes: the original 128 MiB plus
  the exactly authenticated 2,014,910,662-byte interrupted checkpoint. Its own
  output, RAM, wall-time and mutable-scope reservations remain unchanged.
- The original prediction/lineage checks, lossless final contrastive model
  compaction, enumerated corpus retirement and direct producer-wait checks remain
  present. Unchanged global storage accounting must include the retained payload.

`argformer_retirement.qualification.json` records a passing synthetic deletion
inventory test. It used a temporary fixture inside this package and three tiny
checkpoint stand-ins. Actual retirement logic removed the new MLM and contrastive
stand-ins and preserved the stopped checkpoint and original configuration. The
scientific compaction and producer lineage were mocked for that fixture; no real
production file was deleted. An initial fixture-only path mismatch is retained
in `retirement_test.log`; the corrected test result is in
`retirement_test_pass.log`. The runtime source and replacement task did not
change during that test repair.

Final source review also checked corrected ARGformer scoring and TreeLoom's
training-completion binding. Both consume immutable checkpoint/completion refs
and accept the preserved completion paths with nested new checkpoints.
