# Gated complete-parent candidate CLI

ROOT owns all actual admissions, gates and launches. No candidate process has
been launched by this source-development task.

The preflight path is ready. Source references for it are listed in
`candidate_preflight_source_manifest.json`. The separate fit driver is still
being extended for a 512 GiB continuation and should not launch until its final
source review is returned.

Use a fresh request JSON with these fields:

```json
{
  "schema": "treeloom_recovery_candidate_v1",
  "phase": "preflight",
  "job_ref": {"path": "the existing AM_treeloom_fit.json", "sha256": "its actual hash"},
  "observation_barrier_ref": {"path": "the existing observation_barrier.json", "sha256": "its actual hash"},
  "output_directory": "experiments/results/a fresh admitted output path",
  "source_refs": ["copy the actual reference objects from the preflight source manifest"],
  "physical_gpu": 0,
  "gpu_uuid": "the verified physical GPU0 UUID",
  "max_rss_bytes": 137438953472,
  "max_cuda_reserved_bytes": 17179869184,
  "minimum_free_gpu_bytes": 19327352832,
  "max_output_bytes": 67108864,
  "max_seconds": 7200,
  "holds_per_block": 128,
  "haplotypes_per_gru": 32,
  "gru_chunk_holds": 1024
}
```

The request example is descriptive; references and UUID must be replaced with
actual metadata. Validation does not launch anything:

```bash
. scripts/project_env.sh
PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python -B -m research.repairs.benchmark_recovery_20261020.training.candidate_supervisor --request REQUEST_PATH --validate-only
```

After ROOT admission, the explicit launch command is the same without
`--validate-only`. The supervisor creates a fresh output, starts one directly
owned worker with visible arguments, monitors RSS/output/wall limits, and records
its genuine OS wait. It can signal only that worker through its retained pidfd.
No retry, source loader, observation disk cache, old-worker attachment, or
unrelated process operation exists in this path.

The worker first writes `worker_identity.json`, then `ready_for_gpu.json`.
It waits without CUDA initialization for ROOT to create `gpu_gate.json` inside
that output. The gate is an object with:

- `schema`: `treeloom_candidate_gate_v1`
- `gate`: `gpu_gate.json`
- `allow`: true
- `request_ref`: exact request path/hash
- `worker_identity`: exact `pid` and `start_ticks` from the worker record
- `gpu_uuid`: verified physical GPU0 UUID
- `max_cuda_reserved_bytes`: 17179869184
- `expires_unix`: a finite future Unix timestamp
- `admission_ref`: a path/hash to ROOT's immutable shared-GPU admission record

The admission record must contain the same `request_ref`, `worker_identity`,
`physical_gpu: 0`, `gpu_uuid`, and `max_cuda_reserved_bytes`. ROOT creates it only
after its shared scheduler has reserved that allocation. The candidate does not
claim to hold an independent global GPU lease.

After gate acceptance the worker verifies the physical GPU0 UUID and available
memory, sets its 16 GiB allocator cap, and runs forward and backward through both
GRU directions at the maximum existing TRAIN sequence length and 32 haplotypes.
Each call uses at most 1024 contiguous holds. Hidden state remains in the graph
across chunks. The model has the original 32,132 parameters. No optimizer step,
actual parent preparation, truth read, ancestry evaluation or reduced-length
stand-in is performed. A successful result is `preflight_passed.json`; ROOT must
also inspect `supervisor_result.json` for actual exit 0.

The fit request uses `phase: fit`, adds `preflight_ref` and `preflight_wait_ref` (actual successful supervisor wait), and pins the final
`candidate_fit.py` in addition to the preflight source closure. Preflight binding
uses a numerical-source identity, so adding a reviewed orchestration module does
not invalidate the tested numerical implementation. The fit worker prepares the
first true epoch-order parent on CPU, waits for its own GPU gate, and retains the
first genuine update for continuation.

Continuation additionally requires `continue_gate.json`, bound to its exact
`qualification_ref`, with `allow_updates_through: 600`, `max_rss_bytes: 549755813888`,
`observation_cache_bytes: 377957122048`, `training_headroom_bytes: 137438953472`,
and `cpu_admission_ref`. The CPU admission record binds the same request and
worker identity plus the 512 GiB process reservation. This is a separate ROOT
capacity decision; initial qualification remains bounded to 128 GiB.
