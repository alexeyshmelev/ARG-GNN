"""Thin ROOT-owned current-Census bridge for the unchanged full30Mb simulator.

No new quota, accounting, lifecycle, admission or retry mechanism is introduced.
The exact original scientific execute() and Writer methods remain unchanged.
A separately reviewed actual ROOT context owns admission and process supervision.
Default invocation is inert. No numerical import occurs before actual context.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE = "research/reviews/frozen_benchmark_training_ready_20261017_v1/train001_simulate_v2/american_full30mb_simulator_bridge_v1.py"
GENERATOR = "src/arggnn/american_c3_simulate_v1.py"
COMMON = "src/arggnn/american_c3_common_v1.py"
CENSUS_CHILD = dict(
    path="research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py",
    sha256="72f6dc51844b2a261b581f5e69bd2a22982e6244c059f24549747a7b851faeeb",
)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def local(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    need(p != ROOT and p.is_relative_to(ROOT) and ".." not in p.parts
         and p.resolve() == p, "Canonical contained repository path")
    return p


def metadata_ref(value):
    p = local(value)
    s = p.lstat()
    need(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_size <= 4 * 2**20,
         "Small regular immutable source or JSON metadata")
    return dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(record):
    need(isinstance(record, dict) and set(record) == {"path", "sha256"}
         and metadata_ref(record["path"]) == record, "Exact immutable source/metadata ref")
    return json.loads(local(record["path"]).read_text())


class _OriginalWriterCurrentGuard:
    """Delegate every original Writer boundary to the unchanged current guard."""
    def __init__(self, guard, resource_check):
        self.guard = guard
        self.resource_check = resource_check

    def budget_check_v9(self, *args, **kwargs):
        self.resource_check("before_original_Writer_current_Census_boundary")
        result = self.guard.budget_check_v9(*args, **kwargs)
        self.resource_check("after_original_Writer_current_Census_boundary")
        return result


def execute_in_admitted_context_v1(config_ref, *, spec, current_guard,
                                    current_guard_source_ref, resource_check):
    """Direct callable for an existing ROOT-owned worker after its actual gate.

    spec/current_guard are the actual current-Census admission and exact child.
    ROOT's guard validates the complete existing authority, storage, source,
    config and runtime binding. The original Writer calls the same guard before
    each planned write and after each serialization. No old adapter is loaded.
    """
    need(Path.cwd() == ROOT and sys.dont_write_bytecode and sys.flags.no_user_site
         and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
         "Exact repository CPU-only -B -s interpreter")
    cfg = read(config_ref)
    need(current_guard_source_ref == CENSUS_CHILD
         and metadata_ref(CENSUS_CHILD["path"]) == CENSUS_CHILD
         and callable(resource_check)
         and callable(getattr(current_guard, "validate_storage_v9", None))
         and callable(getattr(current_guard, "budget_check_v9", None)),
         "Actual immutable current-Census child guard; no legacy fallback")
    need(spec.get("root_authorizer") == "root_agent"
         and spec.get("execution_authorized") is True
         and spec.get("GPU_authorized") is False
         and spec.get("source_ref") == metadata_ref(SOURCE)
         and spec.get("config_ref") == config_ref
         and cfg.get("execution_enabled") is True
         and cfg.get("sequence_length_bp") == 30000000,
         "Genuine ROOT CPU source/config authority for one complete30Mb parent")
    registry = read(metadata_ref(
        "research/reviews/ARGformer_American_C3_implementation_20261015_v1/parent_seed_registry_inactive.json"))
    matches = [r for r in registry["parents"] if r["parent_id"] == cfg.get("parent_id")]
    need(len(matches) == 1 and matches[0]["split"] == cfg.get("split") == "train"
         and matches[0]["parent_id"] != "american_c3_train_000"
         and matches[0]["ancestry_seed"] == cfg.get("ancestry_seed")
         and matches[0]["mutation_seed"] == cfg.get("mutation_seed"),
         "Original full-length missing TRAIN parent only; do not rerun completedTRAIN000")
    template = read(metadata_ref(
        "research/reviews/ARGformer_American_C3_implementation_20261015_v1/simulation_full30mb_inactive.json"))
    mutable = {"parent_id", "split", "ancestry_seed", "mutation_seed", "execution_enabled"}
    need(set(cfg) == set(template) and all(cfg[k] == v for k, v in template.items() if k not in mutable),
         "Only original registered parent identity/seeds and actual activation may differ")
    required = [metadata_ref(p) for p in (SOURCE, GENERATOR, COMMON)]
    required += [config_ref, spec["runtime_ref"], current_guard_source_ref]
    bindings = spec["storage"]["reservation"]["consumer_bindings"]
    need(all(r in bindings for r in required), "Exact real source/config/runtime/current-child direct bindings")
    resource_check("before_original_simulator_current_Census_validation")
    # Guard authority covers the original primary plus separate charged AUX.
    # This operational view is never passed to the scientific simulator.
    reserved=spec["storage"]["reservation"]["reserved_bytes"]
    need(type(reserved) is int and reserved==spec["storage"]["reserved_new_bytes"]
         and reserved>=cfg["output_cap_bytes"],
         "Exact admitted whole-scope cap; original primary cap remains unchanged")
    storage_cfg=dict(cfg,output_cap_bytes=reserved)
    current_guard.validate_storage_v9(
        spec, storage_cfg, spec["storage"], source_ref=spec["source_ref"],
        config_ref=config_ref, runtime_ref=spec["runtime_ref"])
    # These original imports do not select/configure any storage adapter.
    from arggnn import american_c3_simulate_v1 as simulator
    from arggnn.american_c3_common_v1 import Writer, runtime_report
    runtime = read(spec["runtime_ref"])
    need(runtime.get("prefix") == str(Path(sys.prefix).resolve())
         and runtime.get("CUDA_VISIBLE_DEVICES") == "",
         "Actual ROOT-qualified original scientific CPU interpreter")
    if "packages" in runtime:
        actual = runtime_report()["packages"]
        need(all(actual.get(k) == v for k, v in runtime["packages"].items()),
             "Exact qualified scientific package metadata")
    resource_check("before_original_full30Mb_scientific_execute")
    writer = Writer(spec, _OriginalWriterCurrentGuard(current_guard, resource_check))
    result = simulator.execute(cfg, writer)
    resource_check("after_original_full30Mb_scientific_execute")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--root-request")
    p.add_argument("--expected-root-request-sha256")
    a = p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,
                              numerical_import=False,simulation=False)))
        return 0
    need(a.root_request and a.expected_root_request_sha256,
         "Actual separately reviewed ROOT request is required")
    request_ref = dict(path=a.root_request, sha256=a.expected_root_request_sha256)
    request = read(request_ref)
    need(request.get("schema") == "frozen_American_full30Mb_simulator_ROOT_request_v1"
         and request.get("root_authorizer") == "root_agent"
         and request.get("execution_authorized") is True
         and request.get("bridge_ref") == metadata_ref(SOURCE)
         and request.get("current_guard_source_ref") == CENSUS_CHILD,
         "Actual one-parent ROOT authority and exact current guard source")
    # This source is supplied/authenticated by ROOT. It must consume the real
    # startup gate, validate the actual admission and retain unchanged owned
    # process/wall/RSS/CPU/storage enforcement before returning this context.
    context_ref = request["root_context_module_ref"]
    need(metadata_ref(context_ref["path"]) == context_ref,
         "Exact independently reviewed ROOT lifecycle context")
    loader = importlib.util.spec_from_file_location("_frozen_American30_ROOT_context",
                                                   local(context_ref["path"]))
    context_module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(context_module)
    ctx = context_module.build_context_v1(request_ref, request)
    need(ctx.get("actual_owned_startup_gate_consumed") is True
         and ctx.get("actual_current_admission_validated") is True
         and ctx.get("outer_owned_limits_enforced") is True,
         "Actual ROOT supervised ownership and unchanged resource limits")
    result = execute_in_admitted_context_v1(
        request["config_ref"], spec=ctx["spec"], current_guard=ctx["current_guard"],
        current_guard_source_ref=request["current_guard_source_ref"],
        resource_check=ctx["resource_check"])
    print(json.dumps(dict(completion_ref=result,root_outer_wait_not_asserted=True)),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
