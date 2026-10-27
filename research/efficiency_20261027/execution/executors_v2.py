"""Level-local diagnostic with only consumed block fields transferred.

The earlier frozen candidate remains evidence. This version removes the unused
upward/downward topology tuple transfers that its inherited _block performed in
addition to the new single packed-plan transfer. It changes no tree equations.
"""
from types import SimpleNamespace
import torch
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import need
from .executors import LevelLocalStagedExecution,build_executor as previous_factory


class MinimalTransferLevelExecution(LevelLocalStagedExecution):
    def _block(self,z):
        a,b,lo,hi,parent,*_ = z
        bank=self.bank
        return SimpleNamespace(
            parent=torch.tensor(parent,dtype=torch.long,device=self.device),
            node_time=bank.node_time[lo:hi].to(self.device),
            sample_index=(bank.sample_index[a:b]-lo).to(self.device),
            edge_geometry=bank.edge_geometry[lo:hi].to(self.device),
            branch_support_true=bank.branch_support_true[lo:hi].to(self.device),
            reference_class=bank.reference_class.to(self.device),
            sample_age=bank.sample_age.to(self.device),
            spans=bank.spans[a:b].to(self.device))

    def sweep_plan_snapshot(self):
        result=super().sweep_plan_snapshot()
        result.update(unused_original_topology_transfers=False,
                      packed_route_transfers_per_sweep=1)
        return result


def build_executor(kind,model,bundle,metadata,*,execution="staged",layout_cache=None,
                   frozen_module=None,**kwargs):
    if kind != "level_local_v2":
        return previous_factory(kind,model,bundle,metadata,execution=execution,
            layout_cache=layout_cache,frozen_module=frozen_module,**kwargs)
    need(execution == "staged", "Minimal transfer diagnostic supports staged execution")
    return MinimalTransferLevelExecution(model,bundle,metadata,layout_cache=layout_cache,
        frozen_module=frozen_module,**kwargs)
