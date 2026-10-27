"""Optional categorical/layout plus windowed-history arm for isolated profiles.

This keeps the original tree sweep. It does not silently include the level-local
sweep, which has not demonstrated a GPU throughput improvement.
"""
from functools import partial
import torch
from research.scalability_20261024.integration.executors import IntegratedStagedExecution
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import _cp,profiled,need
from .executors import build_executor as sweep_factory
from .history_pool import stable_leaf_history


class WindowedHistoryExecution(IntegratedStagedExecution):
    @profiled
    def context_forward(self):
        self.tick("bounded_complete_parent_begin")
        pairs=[_cp(partial(self._first,z)) for z in self.layout]
        up=tuple(x[0] for x in pairs)
        out=tuple(x[1] for x in pairs)
        eligible=self._finite_own_episode(up)
        forces=[_cp(partial(self._forcing,z),u,o) for z,u,o in zip(self.layout,up,out)]
        force=torch.cat(forces)
        del forces,out,pairs
        ell=self.model.memory.ell_theta.to("cpu")
        value=stable_leaf_history(self.model.memory,force,ell,self.bank.spans,self.bundle.cut,
            device=self.device,holds_per_window=self.history_window,
            resource_check=self.tick,profile=self.profile)
        del force
        sizes=[self.bundle.cut.group_offsets[z[1]]-self.bundle.cut.group_offsets[z[0]] for z in self.layout]
        values=value.split(sizes,dim=0)
        contexts=[_cp(partial(self._second,z),u,v,eligible) for z,u,v in zip(self.layout,up,values)]
        context=torch.cat(contexts,dim=0)
        del contexts,up,values,value
        self.tick("bounded_complete_parent_context")
        return context


def build_executor(kind,model,bundle,metadata,*,execution="staged",layout_cache=None,
                   frozen_module=None,**kwargs):
    if kind != "history_composed":
        return sweep_factory(kind,model,bundle,metadata,execution=execution,
            layout_cache=layout_cache,frozen_module=frozen_module,**kwargs)
    need(execution == "staged", "History diagnostic supports original staged execution only")
    kwargs.pop("max_sweep_plan_bytes",None)
    kwargs.pop("sweep_plan_admit",None)
    return WindowedHistoryExecution(model,bundle,metadata,layout_cache=layout_cache,
        frozen_module=frozen_module,**kwargs)
