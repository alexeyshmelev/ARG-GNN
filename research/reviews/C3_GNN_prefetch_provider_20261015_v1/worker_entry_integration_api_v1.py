"""Exact optional worker-entry call APIs; no launcher or execution authority.

The root successor must bind this source and all caps to its actual admitted
request. Existing current980 zero-worker sources remain unchanged.
"""
from __future__ import annotations
import inspect
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAP_KEYS = frozenset(("num_workers", "max_block_bytes", "worker_rss_cap_bytes",
                     "aggregate_rss_cap_bytes", "task_timeout_seconds", "drain_timeout_seconds"))


def bootstrap_before_torch_v1():
    """Call after root authorization, before the first Torch/PyG import."""
    from .abstract_ipc_bootstrap_v1 import ensure_abstract_ipc_before_torch_v1
    return ensure_abstract_ipc_before_torch_v1()


def make_observation_provider_factory_v1(*, operational_caps, resource_check):
    """Call after bootstrap; caps must come from the actual root-bound request."""
    if not isinstance(operational_caps, dict) or set(operational_caps) != CAP_KEYS:
        raise ValueError("Exactly the six explicit admitted observation-prefetch caps are required")
    if not callable(resource_check):
        raise ValueError("Root runtime callback required")
    bootstrap_before_torch_v1()  # Rejects missing/late or changed bootstrap.
    from .prefetch_provider_v1 import ObservationProviderFactoryV1
    from arggnn.american_c3_common_v1 import ref
    factory = ObservationProviderFactoryV1(**dict(operational_caps), resource_check=resource_check)
    # Preserve this adapter as part of the optional protocol binding as well.
    factory.execution_binding["source_refs"].append(ref(str(Path(__file__).relative_to(ROOT))))
    return factory


def run_train_stage_with_prefetch_v1(cfg, *, hooks, observation_prefetch_caps,
        parent_plan_refs, actual_train_bindings, writer, device, worker_tmp_directory,
        resolve_train_truth, runtime_guard, resume_ref=None):
    """Root successor replacement for its train-stage call only.

    Requires the reviewed/unapplied trainer and hook factory patches. No fallback
    to an old API, model alteration, unbound caps file or monkeypatch is provided.
    """
    stage = getattr(hooks, "run_train_stage_v1", None)
    if not callable(stage) or "observation_provider_factory" not in inspect.signature(stage).parameters:
        raise ValueError("Explicit patched run_train_stage_v1 observation-provider API required")
    factory = make_observation_provider_factory_v1(
        operational_caps=observation_prefetch_caps, resource_check=runtime_guard)
    return stage(cfg, parent_plan_refs=parent_plan_refs,
        actual_train_bindings=actual_train_bindings, writer=writer, device=device,
        worker_tmp_directory=worker_tmp_directory, resolve_train_truth=resolve_train_truth,
        runtime_guard=runtime_guard, resume_ref=resume_ref,
        observation_provider_factory=factory)
