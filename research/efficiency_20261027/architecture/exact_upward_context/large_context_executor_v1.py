"""Separate single-use released-upward/model-device-context512 composition."""
from research.efficiency_20261027.execution.large_block_executors import _LargeInitialization
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import need
from .memory_v1 import DEFAULT_SCRATCH_BYTES
from .upward_context_v1 import UpwardContextExecution


class LargeUpwardContextExecution(_LargeInitialization, UpwardContextExecution):
    pass


def build_executor(model, bundle, metadata, *, resident_scratch_bytes=DEFAULT_SCRATCH_BYTES, resident_soft_bytes=74*2**30, **kwargs):
    need(kwargs.get('holds_per_block', 512) == 512,
         'Residency candidate retains the qualified512-hold partition')
    need(type(resident_scratch_bytes) is int and resident_scratch_bytes >= DEFAULT_SCRATCH_BYTES,
         'Explicit scratch allowance cannot reduce the sixteen-GiB floor')
    need(type(resident_soft_bytes) is int and 0<resident_soft_bytes<=74*2**30,'Explicit ROOT budget at most74GiB')
    kwargs['holds_per_block'] = 512
    executor = LargeUpwardContextExecution(model, bundle, metadata, **kwargs)
    executor.resident_scratch_bytes = resident_scratch_bytes
    executor.resident_soft_bytes = resident_soft_bytes
    return executor
