# Actual Torch/PyG CPU transport receipt and final integration API

The single requested actual-tensor fixture passed on 2026-10-15 with Python 3.11.17, Torch 2.6.0+cu124 and PyG 2.7.0 in the existing .runtime/env. No model, optimizer, scientific data, GPU context, project guard, admission, accounting or Git operation was used.

torch_pyg_transport_fixture_v1.py exercised the frozen bounded_workers_v1.py and fresh abstract_ipc_bootstrap_v1.py. It transported actual PyG Data objects containing scalar, empty and nonempty Torch tensors. Every nonempty received tensor used actual CPU shared storage and matched its tiny literal source payload exactly.

The one scenario had normal and single injected-failure phases, each with two spawned workers and at most two in-flight tasks. Four normal payloads and one successful prefix before the injected failure each contained only 84 tensor bytes. The normal workers exited [0, 0]. The injected failure was propagated and both owned workers were joined after termination, with actual exits [-15, -15]. No fallback or retry occurred.

Both normal workers' actual PyTorch resource-sharer listeners were observed using leading-NUL abstract AF_UNIX addresses in the new c3-gnn-prefetch-v1 namespace. No filesystem socket nodes appeared anywhere in the fixture TMP namespace. CUDA_VISIBLE_DEVICES was empty before Torch/PyG imports and CUDA remained uninitialized. The fixture TMP directories are preserved.

Actual outputs:
- torch_pyg_transport_receipt.v1.json contains payload/shared-storage checks, runtime, exact worker PIDs/exits and IPC evidence.
- torch_pyg_transport_actual_wait.v1.json records the actual direct outer command session 37059 completion with exit 0 as observed by root's assigned source worker.
- transport_handoff.v1.json freezes this supplement and the previously frozen provider/core/bootstrap/source handoff.

This qualifies the tiny actual PyG/shared-tensor transport mechanism, normal shutdown and injected-failure drain in that runtime. It does not establish real Dataset payload coverage, complete-parent training memory, gradient equivalence, throughput, speedup or production readiness. No further numerical campaign was run.

## Exact worker-entry API

worker_entry_integration_api_v1.py provides three callable APIs. It has no launcher, CLI, admission or authority.

Before the first Torch/PyG import in a separately authorized root successor:

```python
from research.reviews.C3_GNN_prefetch_provider_20261015_v1.worker_entry_integration_api_v1 import (
    bootstrap_before_torch_v1,
    run_train_stage_with_prefetch_v1,
)
ipc_receipt = bootstrap_before_torch_v1()
```

After the original admitted runtime/title/device setup, replace only the train-stage call with:

```python
result = run_train_stage_with_prefetch_v1(
    config,
    hooks=h,
    observation_prefetch_caps=actual_admitted_prefetch_caps,
    parent_plan_refs=b["parent_plan_refs"],
    actual_train_bindings=b["actual_train_bindings"],
    writer=w,
    device=device,
    worker_tmp_directory=cfg["short_tmp_directory"],
    resolve_train_truth=train_truth,
    runtime_guard=fast,
    resume_ref=None,
)
```

actual_admitted_prefetch_caps must contain exactly these six keys, bound by root's actual future request/source closure:
num_workers, max_block_bytes, worker_rss_cap_bytes, aggregate_rss_cap_bytes, task_timeout_seconds, drain_timeout_seconds.

There is no separate unbound caps-file option. CPU_observation_cache_bytes remains in the existing scientific config and must cover max_block_bytes. That block cap must cover the maximum feature planning bound across the frozen original plans. The current main-process-only workers=0 callback is not a claim of aggregate worker qualification; the new provider adds exact owned worker PID/start RSS checks, and root must authorize the aggregate caps and process supervision in the successor.

For a caller that needs the factory directly:

```python
from research.reviews.C3_GNN_prefetch_provider_20261015_v1.worker_entry_integration_api_v1 import make_observation_provider_factory_v1
factory = make_observation_provider_factory_v1(
    operational_caps=actual_admitted_prefetch_caps,
    resource_check=fast,
)
```

Its constructor API stays:
factory(plan_ref, *, parent_id, worker_tmp_directory, cache_bytes).

The reviewed unapplied patch adds observation_provider_factory=None to both:
- run_train_stage_v1(cfg, *, parent_plan_refs, actual_train_bindings, writer, device, worker_tmp_directory, resolve_train_truth, runtime_guard, resume_ref=None, observation_provider_factory=None).
- run_train_v1(cfg, *, parent_plan_refs, writer, device, worker_tmp_directory, resolve_train_truth, runtime_guard, resume_ref=None, train_target_joiner_factory=None, observation_provider_factory=None).

The adapter requires the explicit patched hook parameter. The factory's actual source refs, adapter source and bounds enter the optional protocol signature. The full parent step, targets, all-20 observation barrier and source checks remain unchanged.

provider_factory.integration.unapplied.patch and root_entry.pre_torch_bootstrap.unapplied.patch remain unapplied. Root must create and pin a separate actual executable successor, include every new provider/core/bootstrap/adapter file and patched source in its transitive closure, bind real caps and retain actual launcher/process-tree accounting. The old current980 source pins remain unchanged.
