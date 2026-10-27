"""Exact existing large-block initializer plus the isolated boundary candidate."""
from research.efficiency_20261027.execution.large_block_executors import _LargeInitialization
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import need
from .boundary_fusion import BoundaryFusedExecution


class LargeBoundaryFusedExecution(_LargeInitialization, BoundaryFusedExecution):
    pass


def build_executor(model, bundle, metadata, **kwargs):
    """Same options/cache/frozen-module contract as selected explicit CPU512."""
    need(kwargs.get('holds_per_block', 512) == 512,
         'This qualification candidate selects the existing 512-hold partition')
    kwargs['holds_per_block'] = 512
    return LargeBoundaryFusedExecution(model, bundle, metadata, **kwargs)
