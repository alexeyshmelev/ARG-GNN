# TreeLoom candidate version 2

ROOT owns all real worker launches, shared GPU0 admissions, gates and actual wait
collection. The worker has no automatic retry. Historical supervisor version 1,
its failure, and the original preflight manifest remain immutable.

Use the pinned project interpreter from the repository root:

```bash
. scripts/project_env.sh
PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python -B -m research.repairs.benchmark_recovery_20261020.training.candidate_supervisor_v2 --request REQUEST_PATH --validate-only
```

Removing `--validate-only` launches one direct worker. The supervisor uses only
its owned `Popen` object for termination and real waits. This project Python has
neither `os.pidfd_open` nor `signal.pidfd_send_signal`. The actual CPU lifecycle
smoke records wait 0 for the successful child and wait -15 for controlled
termination, with both exact identities absent afterwards.

Use `candidate_preflight_source_manifest_v2.json` for preflight and
`candidate_fit_source_manifest_v1.json` for fitting. Both retain the historical
`candidate_supervisor.py` reference required by the frozen common validator and
add the live `candidate_supervisor_v2.py` reference. Their numerical identity is
unchanged. The fit manifest also pins the fit driver, compact RAM codec, integer
index implementation and metadata bounds.

The request and gate schemas remain as described in `CANDIDATE_CLI.md`, with
these additional version 2 request fields:

- `minimum_free_gpu_bytes`: at least 27917287424 (26 GiB), checked by the worker
  after the actual ROOT GPU gate and before CUDA initialization.
- `shared_gpu_free_floor_bytes`: exactly 8589934592 (8 GiB). After a GPU gate
  exists, the supervisor checks physical GPU0 UUID/free memory at intervals of
  at least two seconds. Three consecutive observations below the floor stop
  only its directly owned test worker. A healthy observation resets the count.

The test does not acquire or claim inheritance of the global GPU lease. ROOT
supplies a separate logical shared admission, preserving the live native job.
The supervisor output records its actual exit code, owned-child absence,
resource violation, GPU floor observations and no-retry status.

## Fit continuation

The worker imports the static `ReadableTreeLoom` runtime and validates the exact
original 20-parent observed barrier and plans before preparation. It prepares
only the first parent in the original epoch order under the initial 128 GiB
RSS limit. After the GPU gate, it runs one complete-parent optimizer update
with all 10000 labels, saves `qualification_step_001.pt`, and waits while
retaining that parent's prepared bundle and the same model and optimizer.

ROOT may then supply the existing qualification-bound continuation gate with
512 GiB process RSS, 352 GiB compact observation cache and 128 GiB working
reserve. The actual original metadata bounds sum to 373741732426 bytes,
including the outer focal tensors, below the 377957122048-byte cache cap.
This is a metadata upper bound, not a measured process peak.

The driver packs the retained first parent without preparing it again. The
remaining epoch-one parents prepare once, then all 20 parents restore from
independent compact RAM records for the remaining 29 epochs. There is no disk
observation cache, eviction or cold reprepare. The final assertions require
exactly 20 preparations, 580 restores, 600 updates and all 49 AdamW counters at
600. The first-step checkpoint and epochs 5, 10, 15, 20, 25 and 30 remain within
the 64 MiB output cap. Selection remains fixed at epoch 30.

Checkpoint and completion schemas preserve the maintained prediction
interface, while recording the candidate request and continuation provenance.
The CPU smoke ran the actual driver with tiny synthetic forests and the actual
RAM codec, validated all 600 synthetic optimizer updates and seven checkpoints,
and passed the maintained saved-checkpoint and completion validators. It loaded
no original observations or truth and initialized no CUDA. It is a control-flow
and lossless-codec check, not scientific training or an accuracy measurement.

## Receipts

- `candidate_supervisor_v2_smoke.result.json`: real direct-child wait/absence.
- `candidate_fit_cpu_test.result.json`: actual fit-driver synthetic CPU run.
- `candidate_preflight_source_manifest_v2.json`: preserved numerical closure.
- `candidate_fit_source_manifest_v1.json`: fit closure and original metadata check.

The parent ROOT chat owns real preflight results, first-parent qualification,
continuation admission, resource scheduling, commits and reporting.
