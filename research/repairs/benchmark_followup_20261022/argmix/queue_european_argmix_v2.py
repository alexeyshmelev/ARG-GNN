"""Archive-only PATH correction, delegating unchanged GPU5 C4 scientific stages."""
import os

from research.repairs.benchmark_resume_20261021.native.c4 import queue_european_argmix_v1 as original
from arggnn.benchmarking import european_argmix_release as release

MODULE = "research.repairs.benchmark_followup_20261022.argmix.queue_european_argmix_v2"
SOURCE_PATH = "research/repairs/benchmark_followup_20261022/argmix/queue_european_argmix_v2.py"
STAGES = original.STAGES


def archive_runner(bounded_runner):
    """Supply the admitted Python lookup path to the unchanged Rust child."""
    release.m._need(callable(bounded_runner), "Actual ROOT-owned bounded native runner required")

    def run(argv, *, cwd, env, limits, log_path):
        runtime_bin = release.native.PREFIX / "bin"
        python = runtime_bin / "python"
        release.m._need(python.is_file() and os.access(python, os.X_OK),
            "Admitted native runtime python executable required")
        child_env = dict(env)
        inherited_path = child_env.get("PATH", os.environ.get("PATH", os.defpath))
        child_env["PATH"] = str(runtime_bin) + (os.pathsep + inherited_path if inherited_path else "")
        return bounded_runner(argv, cwd=cwd, env=child_env, limits=limits, log_path=log_path)

    return run


def make_call(stage, arguments, output_directory, *, python):
    call = original.make_call(stage, arguments, output_directory, python=python)
    call["module"] = MODULE
    call["source_refs"].append(release.m._ref(release.native.ROOT / SOURCE_PATH))
    return call


def execute_stage(spec, *, resource_check, execution_check, bounded_runner=None):
    if spec["stage"] == "release_archive":
        bounded_runner = archive_runner(bounded_runner)
    return original.execute_stage(spec, resource_check=resource_check,
        execution_check=execution_check, bounded_runner=bounded_runner)
