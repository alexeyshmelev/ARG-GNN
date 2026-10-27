"""Separate opt-in packed-input executor; all trunk math remains inherited."""
import json
import torch
from research.efficiency_20261027.execution.large_block_executors import LargeExplicitVJPExecution
from .hot_block_pack_v2 import BlockInputPack,CUDAPackAdmission,need


def provenance(bundle,metadata):
    return json.dumps(dict(parent_id=bundle.parent_id,
        observation_ref=bundle.observation_ref,focal_registry_ref=bundle.focal_registry_ref,
        native_preparation_ref=bundle.native_preparation_ref,
        scope=metadata.get('layout_cache_scope','complete_parent')),sort_keys=True,separators=(',',':'))


class PackedBlockInputs:
    def bind_hot_pack(self,*,provenance,max_payload_bytes,admit,CPU_test_only=False):
        need(not hasattr(self,'_hot_pack'),'One original input pack per executor required')
        self._hot_provenance=provenance
        self._hot_pack=BlockInputPack(self.bank,self.layout,device=self.device,provenance=provenance,
            max_payload_bytes=max_payload_bytes,admit=admit,CPU_test_only=CPU_test_only)

    def _block(self,z):
        return self._hot_pack.block(self.bank,z,provenance=self._hot_provenance)

    def context_forward(self):
        need(torch.is_grad_enabled(),'This opt-in candidate requires one full training backward')
        self._hot_pack.begin_forward()
        try:
            context=super().context_forward()
            need(context.grad_fn is not None,'Original explicit trunk autograd node required')
            pack=self._hot_pack
            # A Node post-hook runs after the complete original custom backward.
            # It returns None and cannot change parameter/context gradients.
            def finished(grad_inputs,grad_outputs):
                pack.backward_complete()
            self._hot_backward_hook=context.grad_fn.register_hook(finished)
            return context
        except BaseException:
            self._hot_pack.abort()
            raise

    def hot_pack_snapshot(self):
        return self._hot_pack.snapshot()

    def release_layout(self):
        pack=getattr(self,'_hot_pack',None)
        if pack is not None:pack.close()
        super().release_layout()

    def abort_hot_pack(self):
        pack=getattr(self,'_hot_pack',None)
        if pack is not None:pack.abort()
        super().release_layout()

    def __del__(self):
        # Destruction discards an unusable graph; it never permits later reuse.
        if getattr(self,'_hot_pack',None) is not None:self._hot_pack.abort()
        super().release_layout()


class PackedLargeExplicitVJPExecution(PackedBlockInputs,LargeExplicitVJPExecution):
    def __init__(self,model,bundle,metadata,*,hot_pack_max_payload_bytes,
                 hot_pack_soft_budget_bytes,hot_pack_scratch_reserve_bytes,**kwargs):
        need(kwargs.get('holds_per_block',512)==512,'This independent candidate selects exactly512hold blocks')
        kwargs['holds_per_block']=512
        super().__init__(model,bundle,metadata,**kwargs)
        admit=CUDAPackAdmission(device=self.device,soft_budget_bytes=hot_pack_soft_budget_bytes,
            scratch_reserve_bytes=hot_pack_scratch_reserve_bytes,resource_check=self.tick)
        self.bind_hot_pack(provenance=provenance(bundle,metadata),
            max_payload_bytes=hot_pack_max_payload_bytes,admit=admit)


def build_executor(model,bundle,metadata,**kwargs):
    return PackedLargeExplicitVJPExecution(model,bundle,metadata,**kwargs)
