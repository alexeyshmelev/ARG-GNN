"""Explicit registry binding in one isolated process, without source execution.

The numerical classes are ordinary imported modules. A process binds exactly one
registry. Historical files remain unchanged. Running workers keep their own
module instances in their separate processes.
"""
from arggnn.benchmarking import treeloom, treeloom_fresh as fresh
from research.reviews.frozen_benchmark_training_ready_20261017_v1 import american_full30mb_frozen_v1 as frozen
from research.reviews.TreeLoom_GPU_history_source_20261017_v1 import bounded_execution_v1 as execution
from . import preparation

_BOUND_REGISTRY = None

class ReadableTreeLoom(fresh.FreshTreeLoom):
    def __init__(self, registry):
        global _BOUND_REGISTRY
        if type(registry) is not fresh.FreshParentRegistry:
            raise ValueError("Authenticated immutable parent registry required")
        if _BOUND_REGISTRY is not None and _BOUND_REGISTRY != registry.reference:
            raise ValueError("A numerical process may bind only one registry")
        treeloom.verify_implementation()
        train = registry.ids("train")
        predictions = registry.ids("validation") + registry.ids("test")
        frozen.TRAIN, frozen.VAL = train, predictions
        preparation.TRAIN, preparation.VAL = train, predictions
        self.registry = registry
        self.frozen = frozen
        self.preparation = preparation
        self.execution = execution
        _BOUND_REGISTRY = dict(registry.reference)
