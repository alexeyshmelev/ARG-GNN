"""Exact inherited trunk equations with separately admitted larger hold blocks.

Only the partition bound and immutable cache type differ from integrated
initialization. All first/forcing/second/site/history and explicit VJP methods
are inherited unchanged. No launcher, monkeypatch or production binding.
"""
from research.benchmark_rebuild_20261024.american.treeloom_speed import staged_execution_v2 as original_staged
from research.efficiency_20261027.execution.large_layout_cache import LargeBlockLayoutCache
from research.efficiency_20261027.execution.explicit_vjp import ExplicitVJPExecution
from research.efficiency_20261027.execution.explicit_resident_vjp import ExplicitResidentVJPExecution


class _LargeInitialization:
    def __init__(self,model,bundle,metadata,*,layout_cache,holds_per_block=1024,
                 haplotypes_per_gru=64,resource_check=lambda label:None,
                 history_holds_per_window=1024,profile=None,frozen_module=None):
        self._initialize(model,bundle,metadata,layout_cache=layout_cache,
            holds_per_block=holds_per_block,haplotypes_per_gru=haplotypes_per_gru,
            resource_check=resource_check,history_holds_per_window=history_holds_per_window,
            profile=profile,frozen_module=frozen_module or original_staged.frozen_default)

    def _initialize(self,model,bundle,metadata,*,layout_cache,holds_per_block,
                    haplotypes_per_gru,resource_check,
                    history_holds_per_window,profile,frozen_module):
        frozen = frozen_module
        need = original_staged.need
        need(type(model) is frozen.AmericanTreeLoomC3FullV1
             and type(bundle) is frozen.AmericanTreeLoomWholeParentV1,
             "Exact original model and authenticated complete bundle required")
        bundle.validate()
        need(bundle.bank.parent.device.type=="cpu"
             and all(x.device.type=="cpu" for x in frozen._bank_tensors(bundle.bank))
             and all(x.device.type=="cpu" for x in bundle.cut.tensors()),
             "Original complete observation bank and recipes remain CPU")
        need(type(holds_per_block) is int and 1<=holds_per_block<=1024
             and type(haplotypes_per_gru) is int and 1<=haplotypes_per_gru<=128
             and callable(resource_check),"Original bounded execution partitions")
        self.model,self.bundle,self.bank = model,bundle,bundle.bank
        self.device = next(model.parameters()).device
        need(self.device.type=="cuda" and self.device.index in (None,0),
             "Only admitted logical cuda0; production GPU guard unchanged")
        import os
        need(os.environ.get("ARGGNN_GPU_MEMORY_POLICY")=="full_device",
             "Original explicit full-device permission required")
        need(isinstance(layout_cache,LargeBlockLayoutCache),
             "Larger executor requires explicitly budgeted LargeBlockLayoutCache")
        self.tick,self.hap_tile = resource_check,haplotypes_per_gru
        self.profile = profile or original_staged.NullProfileV1()
        self.history_window = history_holds_per_window
        counts = metadata["episode_metadata"]["native_nodes_per_tree"]
        need(len(counts)==bundle.native_holds
             and all(type(n) is int and n>=1600 for n in counts)
             and sum(counts)==self.bank.parent.numel(),
             "Original complete native per-hold node partition required")
        self._layout_lease = layout_cache.acquire(bundle,metadata,
            holds_per_block=holds_per_block,resource_check=resource_check)
        self.layout = self._layout_lease.layout


class LargeExplicitVJPExecution(_LargeInitialization, ExplicitVJPExecution):
    pass


class LargeExplicitResidentVJPExecution(_LargeInitialization, ExplicitResidentVJPExecution):
    pass


def build_executor(model,bundle,metadata,*,resident_context=False,**kwargs):
    cls = LargeExplicitResidentVJPExecution if resident_context else LargeExplicitVJPExecution
    return cls(model,bundle,metadata,**kwargs)
