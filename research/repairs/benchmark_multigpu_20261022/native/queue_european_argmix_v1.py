"""C4 per-job GPU2/3/5 bindings with the admitted archive Python PATH correction."""
from research.repairs.benchmark_resume_20261021.native.c4 import queue_european_argmix_v1 as original
from research.repairs.benchmark_followup_20261022.argmix.queue_european_argmix_v2 import archive_runner
from . import c4_runtime_bindings_v1 as bindings

MODULE = "research.repairs.benchmark_multigpu_20261022.native.queue_european_argmix_v1"
STAGES = original.STAGES
SOURCE_PATHS = (
    "research/repairs/benchmark_multigpu_20261022/native/queue_european_argmix_v1.py",
    "research/repairs/benchmark_multigpu_20261022/native/c4_runtime_bindings_v1.py",
    "research/repairs/benchmark_multigpu_20261022/native/device_binding_v1.py",
    "research/repairs/benchmark_multigpu_20261022/native/sitecustomize_multigpu_v1.py",
    "research/repairs/benchmark_followup_20261022/argmix/queue_european_argmix_v2.py",
)


def make_call(stage, arguments, output_directory, *, python):
    call = original.make_call(stage, arguments, output_directory, python=python)
    call["module"] = MODULE
    call["source_refs"].extend(bindings.m._ref(bindings.native.ROOT / path) for path in SOURCE_PATHS)
    return call


def execute_stage(spec, *, resource_check, execution_check, bounded_runner=None):
    stage = spec["stage"]
    if stage == "release_archive":
        bounded_runner = archive_runner(bounded_runner)
    if stage not in ("release_fit", "release_infer"):
        return original.execute_stage(spec, resource_check=resource_check,
            execution_check=execution_check, bounded_runner=bounded_runner)
    arguments = dict(spec["arguments"])
    headroom = arguments.pop("write_headroom_bytes")
    if stage == "release_infer":
        bindings.m._need(callable(bounded_runner), "Actual ROOT-owned bounded native runner required")
        arguments["bounded_runner"] = bounded_runner
    function = bindings.fit_release if stage == "release_fit" else bindings.infer_release
    return function(output_directory=spec["output_directory"], **arguments,
        guard=resource_check, execution_check=execution_check, write_headroom_bytes=headroom)
