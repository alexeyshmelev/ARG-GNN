"""GPU5 C4 successor with the original queue interface and unchanged scientific calls."""
from research.repairs.benchmark_recovery_20261020.scheduler import queue_european_argmix_v1 as original
from . import runtime_bindings_v1 as bindings

MODULE = "research.repairs.benchmark_resume_20261021.native.c4.queue_european_argmix_v1"
STAGES = original.STAGES
SOURCE_PATHS = (
    "research/repairs/benchmark_resume_20261021/native/c4/queue_european_argmix_v1.py",
    "research/repairs/benchmark_resume_20261021/native/c4/runtime_bindings_v1.py",
    "research/repairs/benchmark_resume_20261021/native/c4/sitecustomize_gpu5_v1.py",
    "research/repairs/benchmark_recovery_20261020/scheduler/queue_european_argmix_v1.py",
    ".runtime/official_argmix_abi210_v1/cu129/source/src/train_joint_model.py",
    ".runtime/official_argmix_abi210_v1/cu129/source/src/argmix_cli.py",
    ".runtime/official_argmix_abi210_v1/cu129/source/src/commands/infer.py",
    ".runtime/official_argmix_abi210_v1/cu129/source/src/commands/runtime.py",
    ".runtime/official_argmix_abi210_v1/cu129/source/src/export_torchscript.py",
    ".runtime/official_argmix_abi210_v1/cu129/source/src/run_inference.py",
    ".runtime/official_argmix_abi210_v1/cu129/source/src/run_joint_inference.py",
)


def make_call(stage, arguments, output_directory, *, python):
    call = original.make_call(stage, arguments, output_directory, python=python)
    call["module"] = MODULE
    call["source_refs"].extend(original.m._ref(original.m.ROOT / path) for path in SOURCE_PATHS)
    return call


def execute_stage(spec, *, resource_check, execution_check, bounded_runner=None):
    stage = spec["stage"]
    if stage not in ("release_fit", "release_infer"):
        return original.execute_stage(spec, resource_check=resource_check,
            execution_check=execution_check, bounded_runner=bounded_runner)
    arguments = dict(spec["arguments"])
    headroom = arguments.pop("write_headroom_bytes")
    if stage == "release_infer":
        original.m._need(callable(bounded_runner), "Actual ROOT-owned bounded native runner required")
        arguments["bounded_runner"] = bounded_runner
    function = bindings.fit_release if stage == "release_fit" else bindings.infer_release
    return function(output_directory=spec["output_directory"], **arguments,
        guard=resource_check, execution_check=execution_check, write_headroom_bytes=headroom)
