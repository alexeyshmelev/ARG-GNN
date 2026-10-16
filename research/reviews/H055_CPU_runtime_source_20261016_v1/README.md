# H055 CPU runtime caller — source-only handoff

The only runtime implementation is `runtime_v1.py`. Its import uses the standard library and performs no file read, guard operation, admission, network request, output creation or scientific execution. The only acquisition entry is `run_v1(root_input_ref)`; one fresh Linux process may invoke it once. Root alone supplies the real original V9 storage authority after its normal admission.

This worker did not import or run the caller, original producer, fixture, guards or checker. No genotype row, empirical metric, model, GPU, admission or Git operation ran. Actual cohort count/transfer/RSS/runtime/output fit remains unqualified. The producer is unchanged.

## Frozen references and execution boundary

`FIXED_REFS` in the caller contains exact paths and SHA256 values for the cleared v2 producer, v2 protocol, byte-identical roster, frozen H055 registration snapshot, original guard/policy, original checker/memo/v4, and authentic marker-repair renewal/certificate/repair authority.

- Producer: `research/reviews/Danish_genotype_corroboration_source_20261016_v2/corroboration_v2.py`, SHA256 `fdd145e0a34dc00c30999c5047f5469f92bf75d88703a5783cc7f3f3e1390a36`.
- Protocol: `research/reviews/Danish_genotype_corroboration_source_20261016_v2/protocol.source_frozen.inactive.v2.json`, SHA256 `2e48ece48e69d8600cc6c88649e451ffe0ff5034e2322d499db1ce4d9de5da6a`.
- Registration snapshot: `research/reviews/H055_root_integration_20261016_v1/registered_H055_frozen_before_data.root_v1.json`, SHA256 `bcc9606966a2b2da05c7067eaaaf1ea468ba7e702e237495339b70bf74bc8fac`.

The snapshot is registration evidence with `execution_authorized=false`. The separate root input supplies actual execution authority. There is no mutable decisions-file dependency.

The caller passes the exact original protocol path, a fresh absolute output path, original `root_authorization`, and two callbacks to the original `producer.run_v1`. It does not patch any producer function, change QC/GP/support/estimand, or provide an alternative writer.

## Exact root metadata shapes

Root creates a fresh metadata folder with existing parents. Keep `config.json`, `runtime.json`, `source_closure.json`, and the actual admitted `storage.json` together in that folder. Keep it disjoint from the scientific output directory. This packet creates none of these job files and contains no metadata/admission generator.

Every reference below has exactly `{"path": "relative/project/path", "sha256": "64 lowercase hex"}`. Source/metadata reads authenticate canonical, bounded, singly linked regular files. Let:

- `F` be the caller's unchanged `FIXED_REFS` dictionary.
- `E` be the actual reference to `runtime_v1.py` (the sealed SHA is in `source_validation_receipt.v1.json`).
- `O` be root's genuine complete original full-state `bindings` dictionary, preserving every original key/reference. At startup this must equal `checker.bindings` exactly.
- `J` be a fresh job ID matching `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`.
- `D` be a fresh relative output under `experiments/results/`, with its immediate parent already present.
- `T0,T1` be timezone-aware current issuance/absolute expiry strings with `0 < T1-T0 <= 14400 seconds`. `T1` must precede the genuine renewal's expiry.

`config.json` is exactly:

```python
{
  "schema": "H055_CPU_config_v1",
  "root_authorizer": "root_agent",
  "hypothesis_id": "H055",
  "fixed_refs": F,
  "device": "cpu",
  "physical_gpu": None,
  "cuda_visible_devices": "",
  "max_cuda_reserved_bytes": 0,
  "max_cpu_rss_bytes": 2147483648,
  "global_wall_seconds": 14400,
  "scientific_output_cap_bytes": 16777216,
  "max_metadata_eligible_SNPs": 100000,
  "charged_scope_cap_bytes": 33554432,
  "auxiliary_scopes": [],
  "original_scope_checker_reuse_root_authorized": True
}
```

`runtime.json` is exactly:

```python
{
  "schema": "H055_CPU_runtime_v1",
  "root_authorizer": "root_agent",
  "job_id": J,
  "output_directory": D,
  "created_at_utc": T0,
  "expires_at_utc": T1,
  "device": "cpu",
  "physical_gpu": None,
  "cuda_visible_devices": "",
  "global_wall_seconds": 14400
}
```

Let `C,R,L,S` be the actual refs for config, runtime, source closure and admitted storage respectively. `source_closure.json` is exactly:

```python
{
  "schema": "H055_CPU_source_closure_v1",
  "root_authorizer": "root_agent",
  "bound_refs": dict(F, runtime_entry=E, config=C, runtime=R),
  "original_bindings": O
}
```

Root's genuine reservation must directly bind the unique union, by relative path/SHA256, of:

1. All `O.values()`.
2. All `bound_refs.values()` whose suffix is `.py`, `.sh` or `.json`.
3. `L`.
4. All exact frozen protocol `source_refs` with those same allowed suffixes.

No other direct binding is accepted; ordering is immaterial. Both the actual list and its unique union must fit the unchanged original limit of64. TSV/TXT refs are not directly admitted: the exact roster and cached source-header bytes are authenticated transitively through the pinned protocol/snapshot/closure, and explicitly read/hash-checked before the producer. There is no derived proxy binding.

`storage.json` is the real original storage record from root's authentic admission, preserving all original fields, budget components, reserve event/root authority, initial base/prefix pins and policy references. The caller checks at least:

```python
{
  "kind": "project_budget_v9_storage_record",
  "active": True,
  "consumer_uses_v9_guard": True,
  "accounting_mode": "prospective_980gb_v1",
  "job_id": J,
  "output_directory": D,
  "expires_at_utc": T1,
  "reserved_new_bytes": 33554432,
  "completed_overlay_policy_ref": F["policy"],
  "reservation": {
    "key": "H055/<root chosen fresh key>",
    "output": "/workspace-SR003.nfs2/ARG-GNN/" + D,
    "auxiliary_scopes": [],
    "reserved_bytes": 33554432,
    "consumer_bindings": "<exact union above>",
    "root_admission": "<actual original reserve authority ref>"
  },
  "budget_components": "<actual original implementation/launcher/adapter refs>",
  "initial_ledger_pins": "<actual unchanged original base/prefix pins>"
}
```

Angle-bracket values in this storage illustration stand for real original values; they are not usable authorities. The unchanged original startup guard and own-scope checker validate them. Storage is bound by the externally supplied exact input ref, so it is not inserted into its own reservation or source closure. This avoids a self-referential storage/input hash. No replacement admission or authority is constructed by the worker.

Finally, `input.json` is exactly:

```python
{
  "schema": "H055_CPU_root_input_v1",
  "status": "active",
  "root_authorizer": "root_agent",
  "hypothesis_id": "H055",
  "registered": True,
  "execution_authorized": True,
  "job_id": J,
  "output_directory": D,
  "created_at_utc": T0,
  "expires_at_utc": T1,
  "runtime_entry_ref": E,
  "refs": {"config": C, "runtime": R, "source_closure": L, "storage": S}
}
```

## Original checks, fast callbacks and finite resources

The caller clears CUDA visibility, rejects a process with model/GPU packages already imported, sets a virtual-address soft ceiling no larger than2GiB (a conservative stronger condition than RSS alone), and checks Linux peak RSS against2GiB. The existing stricter address limit is preserved if lower. A process-wide real-time alarm plus callbacks enforce at most4hours and the earlier absolute root expiry. No Torch/model import is needed.

Only after these checks does it install the authentic original renewal, then import/configure the exact original guard, then construct the unchanged `ControllerMetadataScopeV1`. The constructor performs one full original startup guard. The first own-scope preflight is exactly `checker.check(required=16MiB + 512KiB)`.

Each original request/write callback performs constant-size counter, clock, fixed package-membership and RSS checks. It performs no per-SNP file stat, checksum, metadata read or global filesystem scan. The original own-scope check runs before about1MiB projected serialized growth, at the next callback after60seconds, around root receipt writes, and finally with `required=0`. Original source/journal/closure checks and stat-only owned-scope measurement remain in use; no later check claims a refreshed global projection.

The callback counter includes the producer's4096-byte mkdir charge plus every serialized producer and root write. Additional allowance for12 allocation units conservatively covers the finite flat directory/file count within the extra charged16MiB slack. A filesystem unit above1MiB or forecast beyond32MiB fails before payload. Root receipts together are capped at512KiB. The original producer independently enforces its16MiB content limit and100,000-row maximum; the extra charge grants no extra scientific rows.

The scientific output directory must remain absent through all startup checks. Only the original producer Writer creates it. After the original call returns and its genuine completed/summary files are verified, the caller exclusively creates `root_runtime_binding.json` and `root_runtime_receipt.json`. It never overwrites producer `completed.json` or `summary.json`. The root receipt records the successful producer and preceding scope check truthfully; the caller returns success only after the final post-write own-scope check. Failure preserves all original output and cannot produce a root receipt before producer success.

## One root invocation

After independent runtime review, the root-owned tiny v2 fixture qualification, and genuine root admission, run once in a fresh process:

```sh
cd /workspace-SR003.nfs2/ARG-GNN
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -B research/reviews/H055_CPU_runtime_source_20261016_v1/runtime_v1.py --execute --input research/reviews/<fresh-root-metadata-folder>/input.json --expected-input-sha256 <actual-root-input-SHA256>
```

The equivalent API call is `run_v1({"path": "<actual input relative path>", "sha256": "<actual input SHA256>"})` in the same fresh Linux/CPU process. No caller, guard, fixture, admission or scientific run was executed to prepare this packet. Root still owns runtime review/admission and actual cohort execution.
