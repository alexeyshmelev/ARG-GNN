"""Explicit successor dispatch; never installs hooks into an existing runtime."""
from research.benchmark_rebuild_20261024.american.treeloom_speed.runtime_v2 import (
    FreshTreeLoom, _need)
from .cache import LayoutCache
from .executors import build_executor
from .step import train_step


class SelectedFreshTreeLoom(FreshTreeLoom):
    """Construct in a new authorized task interpreter, with explicit cache budget."""
    def __init__(self,registry,*,executor_kind,layout_cache=None):
        _need(executor_kind in ("original","integrated"),
              "Explicit original/integrated selection required")
        _need(executor_kind=="original" or isinstance(layout_cache,LayoutCache),
              "Integrated runtime requires declared additional host-cache reserve")
        super().__init__(registry)
        self.executor_kind = executor_kind
        self.layout_cache = layout_cache

    def train_step(self,model,bundle,targets,optimizer,metadata,*,
                   resource_check,profile=None):
        self._check_bundle(model,bundle,metadata,"train")
        _need(targets.receipt.get("parent_registry_ref")==self.registry.reference
              and targets.receipt.get("split")=="train",
              "Separate fresh registered TRAIN target join required")
        return train_step(model,bundle,targets,optimizer,metadata,
            resource_check=resource_check,frozen_module=self.frozen,
            status_path=self.status_path,holds_per_block=128,
            haplotypes_per_gru=64,history_holds_per_window=1024,profile=profile,
            executor_kind=self.executor_kind,layout_cache=self.layout_cache)

    def predict_logits(self,model,bundle,metadata,*,resource_check,profile=None):
        split = self._check_bundle(model,bundle,metadata)
        _need(split in ("validation","test") and callable(resource_check),
              "Registered observation-only held-out prediction required")
        import torch
        model.eval()
        resource_check("fresh_TreeLoom_prediction")
        with torch.no_grad():
            executor = build_executor(self.executor_kind,model,bundle,metadata,
                execution="bounded",layout_cache=self.layout_cache,
                resource_check=resource_check,holds_per_block=128,
                haplotypes_per_gru=64,history_holds_per_window=1024,profile=profile)
            try:
                return executor.forward()
            finally:
                if hasattr(executor,"release_layout"):
                    executor.release_layout()

    def close_layout_cache(self):
        if self.layout_cache is not None:
            self.layout_cache.clear()
