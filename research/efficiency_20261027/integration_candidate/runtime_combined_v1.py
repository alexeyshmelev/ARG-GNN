"""Explicit successor runtime; original model, observation and prediction APIs."""
from research.benchmark_rebuild_20261024.american.treeloom_speed.runtime_v2 import (
    FreshTreeLoom, _need)
from .memory_v1 import GovernedLayoutCache, GovernedLargeLayoutCache
from .step_combined_v1 import train_step


class ExplicitFreshTreeLoom(FreshTreeLoom):
    def __init__(self,registry,*,layout_cache,holds_per_block):
        _need(holds_per_block == 512 and isinstance(layout_cache, GovernedLargeLayoutCache), 'Actual-RSS governed layout cache required')
        super().__init__(registry)
        self.layout_cache=layout_cache
        self.holds_per_block=holds_per_block

    def train_step(self,model,bundle,targets,optimizer,metadata,*,resource_check,profile=None):
        self._check_bundle(model,bundle,metadata,'train')
        _need(targets.receipt.get('parent_registry_ref')==self.registry.reference
              and targets.receipt.get('split')=='train', 'Original registered TRAIN target join required')
        return train_step(model,bundle,targets,optimizer,metadata,resource_check=resource_check,
            frozen_module=self.frozen,status_path=self.status_path,holds_per_block=self.holds_per_block,
            haplotypes_per_gru=64,history_holds_per_window=1024,profile=profile,
            executor_kind='explicit_vjp',layout_cache=self.layout_cache)

    def close_layout_cache(self):
        self.layout_cache.clear()
