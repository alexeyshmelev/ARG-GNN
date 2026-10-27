"""Separate single-use CPU-context512 release adapter with actual-count admission."""
from research.efficiency_20261027.execution.large_block_executors import _LargeInitialization
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import need
from .memory_v1 import DEFAULT_SCRATCH_BYTES
from .upward_release_v1 import UpwardReleaseExecution


class LargeUpwardReleaseExecution(_LargeInitialization, UpwardReleaseExecution):
    pass


def build_executor(model, bundle, metadata, *, resident_scratch_bytes=DEFAULT_SCRATCH_BYTES, **kwargs):
    need(kwargs.get('holds_per_block', 512) == 512,
         'Residency candidate retains the qualified512-hold partition')
    need(type(resident_scratch_bytes) is int and resident_scratch_bytes >= DEFAULT_SCRATCH_BYTES,
         'Explicit scratch allowance cannot reduce the sixteen-GiB floor')
    kwargs['holds_per_block'] = 512
    executor = LargeUpwardReleaseExecution(model, bundle, metadata, **kwargs)
    executor.resident_scratch_bytes = resident_scratch_bytes
    return executor
